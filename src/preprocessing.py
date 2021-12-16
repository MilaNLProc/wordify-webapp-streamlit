import multiprocessing as mp
import os
import re
import string
from collections import OrderedDict
from typing import Callable, List, Optional, Union

import spacy
import vaex
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from textacy.preprocessing import make_pipeline, normalize, remove, replace

from .configs import Languages

# more [here](https://github.com/fastai/fastai/blob/master/fastai/text/core.py#L42)
# and [here](https://textacy.readthedocs.io/en/latest/api_reference/preprocessing.html)
# fmt: off
_re_normalize_acronyms = re.compile(r"(?:[a-zA-Z]\.){2,}")
def normalize_acronyms(t: str) -> str:
    return _re_normalize_acronyms.sub(t.translate(str.maketrans("", "", string.punctuation)).upper(), t)


_re_non_word = re.compile(r"[^A-Za-z]+")
def remove_non_word(t: str) -> str:
    "Removes non-words characters and digits from the text using the regex `[^A-Za-z]+`"
    return _re_non_word.sub(" ", t)


_re_space = re.compile(r" {2,}")
def normalize_useless_spaces(t: str) -> str:
    return _re_space.sub(" ", t)


_re_rep = re.compile(r"(\S)(\1{2,})")
def normalize_repeating_chars(t: str) -> str:
    def _replace_rep(m):
        c, cc = m.groups()
        return c

    return _re_rep.sub(_replace_rep, t)


_re_wrep = re.compile(r"(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)")
def normalize_repeating_words(t: str) -> str:
    def _replace_wrep(m):
        c, cc, e = m.groups()
        return c

    return _re_wrep.sub(_replace_wrep, t)


_re_remove_numbers = re.compile(r"\d+")
def remove_numbers(t: str) -> str:
    return _re_remove_numbers.sub(" ", t)


def lowercase(t: str) -> str:
    "Lowercases the text"
    return t.lower()


def strip(t: str) -> str:
    return t.strip()


def lemmatize_remove_stopwords(doc: spacy.tokens.doc.Doc) -> str:
    return " ".join(
        [t.lemma_ for t in doc if t.lemma_ != "-PRON-" and not t.is_stop]
    )


def remove_stopwords(doc: spacy.tokens.doc.Doc) -> str:
    return " ".join([t.text for t in doc if not t.is_stop])


def lemmatize_keep_stopwords(doc: spacy.tokens.doc.Doc) -> str:
    return " ".join([t.lemma_ for t in doc if t.lemma_ != "-PRON-"])


def identity(t):
    return t


# fmt: on
class PreprocessingPipeline:
    def __init__(
        self,
        language: str,
        pre_steps: Optional[List[str]],
        lemmatization_step: Optional[str],
        post_steps: Optional[List[str]],
    ):

        self.language = language
        self.pre_steps = pre_steps
        self.lemmatization_step = lemmatization_step
        self.post_steps = post_steps

        self.pre = self.make_pipe_component(self.pre_steps, self.language)
        self.post = self.make_pipe_component(self.post_steps, self.language)
        self.nlp = self.make_nlp(self.lemmatization_step, self.language)
        self.lemma = self.make_lemma(self.lemmatization_step, self.language)

    # def apply_multiproc(fn, series):
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         new_series = pool.map(fn, series)
    #     return new_series

    def vaex_process(self, df: DataFrame, text_column: str) -> DataFrame:
        def fn(t):
            return self.post(self.lemma(self.nlp(self.pre(t))))

        vdf = vaex.from_pandas(df)
        vdf["processed_text"] = vdf.apply(
            fn, arguments=[vdf[text_column]], vectorize=False
        )
        df = vdf.to_pandas_df()

        return df

    # def __call__(self, series: Series) -> Series:
    #     if self.pre:
    #         series = series.map(self.pre)

    #     if self.lemma:
    #         total_steps = len(series) // 100
    #         res = []
    #         pbar = st.progress(0)
    #         for i, doc in enumerate(
    #             self.nlp.pipe(series, batch_size=500, n_process=os.cpu_count())
    #         ):
    #             res.append(self.lemma(doc))

    #             if i % total_steps == 0:
    #                 pbar.progress(1)

    #         series = pd.Series(res)

    #     if self.post:
    #         series = series.map(self.post)

    #     return series

    @classmethod
    def make_pipe_component(cls, steps: Optional[List[str]], language: str) -> Callable:
        if not steps:
            return identity

        elif language in ("MultiLanguage", "Chinese") and "remove_non_words" in steps:
            idx = steps.index("remove_non_words")
            steps = (
                steps[:idx]
                + ["remove_numbers", "remove_punctuation"]
                + steps[idx + 1 :]
            )

        components = [cls.pipeline_components()[step] for step in steps]

        return make_pipeline(*components)

    @staticmethod
    def make_nlp(
        lemmatization_step: Optional[str], language: str
    ) -> Union[spacy.language.Language, Callable]:
        if (
            lemmatization_step is None
            or lemmatization_step == "Disable lemmatizer"
            or (
                lemmatization_step == "Spacy lemmatizer (keep stopwords)"
                and language in ("MultiLanguage", "Chinese")
            )
        ):
            return identity
        return spacy.load(Languages[language].value, disable=["parser", "ner"])

    @classmethod
    def make_lemma(cls, lemmatization_step: Optional[str], language: str) -> Callable:

        if (
            lemmatization_step is None
            or lemmatization_step == "Disable lemmatizer"
            or (
                lemmatization_step == "Spacy lemmatizer (keep stopwords)"
                and language in ("MultiLanguage", "Chinese")
            )
        ):
            return identity

        elif (
            lemmatization_step == "Spacy lemmatizer (remove stopwords)"
            and language in ("MultiLanguage", "Chinese")
        ):
            return cls.lemmatization_component().get("Remove stopwords")

        return cls.lemmatization_component().get(lemmatization_step)

    @staticmethod
    def pipeline_components() -> "OrderedDict[str, Callable]":
        """Returns available cleaning steps in order"""
        return OrderedDict(
            [
                ("lowercase", lowercase),
                ("normalize_unicode", normalize.unicode),
                ("normalize_bullet_points", normalize.bullet_points),
                ("normalize_hyphenated_words", normalize.hyphenated_words),
                ("normalize_quotation_marks", normalize.quotation_marks),
                ("normalize_whitespaces", normalize.whitespace),
                ("replace_urls", replace.urls),
                ("replace_currency_symbols", replace.currency_symbols),
                ("replace_emails", replace.emails),
                ("replace_emojis", replace.emojis),
                ("replace_hashtags", replace.hashtags),
                ("replace_numbers", replace.numbers),
                ("replace_phone_numbers", replace.phone_numbers),
                ("replace_user_handles", replace.user_handles),
                ("normalize_acronyms", normalize_acronyms),
                ("remove_accents", remove.accents),
                ("remove_brackets", remove.brackets),
                ("remove_html_tags", remove.html_tags),
                ("remove_punctuation", remove.punctuation),
                ("remove_non_words", remove_non_word),
                ("remove_numbers", remove_numbers),
                ("normalize_useless_spaces", normalize_useless_spaces),
                ("normalize_repeating_chars", normalize_repeating_chars),
                ("normalize_repeating_words", normalize_repeating_words),
                ("strip", strip),
            ]
        )

    @staticmethod
    def lemmatization_component() -> "OrderedDict[str, Optional[Callable]]":
        return OrderedDict(
            [
                ("Spacy lemmatizer (keep stopwords)", lemmatize_keep_stopwords),
                ("Spacy lemmatizer (remove stopwords)", lemmatize_remove_stopwords),
                ("Disable lemmatizer", identity),
                ("Remove stopwords", remove_stopwords),
            ]
        )
