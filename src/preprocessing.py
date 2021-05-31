import re
import string
from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import spacy
import streamlit as st
from pandas.core.series import Series
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from stqdm import stqdm
from textacy.preprocessing import make_pipeline, normalize, remove, replace

from .configs import Languages

stqdm.pandas()


def encode(text: pd.Series, labels: pd.Series):
    """
    Encodes text in mathematical object ameanable to training algorithm
    """
    tfidf_vectorizer = TfidfVectorizer(
        input="content",  # default: file already in memory
        encoding="utf-8",  # default
        decode_error="strict",  # default
        strip_accents=None,  # do nothing
        lowercase=False,  # do nothing
        preprocessor=None,  # do nothing - default
        tokenizer=None,  # default
        stop_words=None,  # do nothing
        analyzer="word",
        ngram_range=(1, 3),  # maximum 3-ngrams
        min_df=0.001,
        max_df=0.75,
        sublinear_tf=True,
    )
    label_encoder = LabelEncoder()

    with st.spinner("Encoding text using TF-IDF and Encoding labels"):
        X = tfidf_vectorizer.fit_transform(text.values)
        y = label_encoder.fit_transform(labels.values)

    return {
        "X": X,
        "y": y,
        "X_names": np.array(tfidf_vectorizer.get_feature_names()),
        "y_names": label_encoder.classes_,
    }


# more [here](https://github.com/fastai/fastai/blob/master/fastai/text/core.py#L42)
# and [here](https://textacy.readthedocs.io/en/latest/api_reference/preprocessing.html)
# fmt: off
_re_normalize_acronyms = re.compile(r"(?:[a-zA-Z]\.){2,}")
def normalize_acronyms(t):
    return _re_normalize_acronyms.sub(t.translate(str.maketrans("", "", string.punctuation)).upper(), t)


_re_non_word = re.compile(r"\W")
def remove_non_word(t):
    return _re_non_word.sub(" ", t)


_re_space = re.compile(r" {2,}")
def normalize_useless_spaces(t):
    return _re_space.sub(" ", t)


_re_rep = re.compile(r"(\S)(\1{2,})")
def normalize_repeating_chars(t):
    def _replace_rep(m):
        c, cc = m.groups()
        return c

    return _re_rep.sub(_replace_rep, t)


_re_wrep = re.compile(r"(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)")
def normalize_repeating_words(t):
    def _replace_wrep(m):
        c, cc, e = m.groups()
        return c

    return _re_wrep.sub(_replace_wrep, t)

# fmt: on
class TextPreprocessor:
    def __init__(
        self,
        language: str,
        cleaning_steps: List[str],
        lemmatizer_when: str = "last",
        remove_stop: bool = True,
    ) -> None:

        # prepare lemmatizer
        self.language = language
        self.nlp = spacy.load(
            Languages[language].value, exclude=["parser", "ner", "pos", "tok2vec"]
        )
        self.lemmatizer_when = self._lemmatization_options().get(lemmatizer_when, None)
        self.remove_stop = remove_stop
        self._lemmatize = self._get_lemmatizer()

        # prepare cleaning
        self.cleaning_steps = [
            self._cleaning_options()[step]
            for step in cleaning_steps
            if step in self._cleaning_options()
        ]
        self.cleaning_pipeline = (
            make_pipeline(*self.cleaning_steps) if self.cleaning_steps else lambda x: x
        )

    def _get_lemmatizer(self) -> Callable:
        """Return the correct spacy Doc-level lemmatizer"""
        if self.remove_stop:

            def lemmatizer(doc: spacy.tokens.doc.Doc) -> str:
                """Lemmatizes spacy Doc and removes stopwords"""
                return " ".join(
                    [t.lemma_ for t in doc if t.lemma_ != "-PRON-" and not t.is_stop]
                )

        else:

            def lemmatizer(doc: spacy.tokens.doc.Doc) -> str:
                """Lemmatizes spacy Doc"""
                return " ".join([t.lemma_ for t in doc if t.lemma_ != "-PRON-"])

        return lemmatizer

    @staticmethod
    def _lemmatization_options() -> Dict[str, str]:
        return {
            "Before preprocessing": "first",
            "After preprocessing": "last",
            "Never! Let's do it quick and dirty": None,
        }

    def lemmatizer(self, series: pd.Series) -> pd.Series:
        """
        Apply spacy pipeline to transform string to spacy Doc and applies lemmatization
        """
        res = []
        pbar = stqdm(total=len(series))
        for doc in self.nlp.pipe(series, batch_size=500):
            res.append(self._lemmatize(doc))
            pbar.update(1)
        pbar.close()
        return pd.Series(res)

    @staticmethod
    def _cleaning_options():
        """Returns available cleaning steps in order"""
        return OrderedDict(
            [
                ("lower", lambda x: x.lower()),
                ("normalize_unicode", normalize.unicode),
                ("normalize_bullet_points", normalize.bullet_points),
                ("normalize_hyphenated_words", normalize.hyphenated_words),
                ("normalize_quotation_marks", normalize.quotation_marks),
                ("normalize_whitespace", normalize.whitespace),
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
                ("normalize_useless_spaces", normalize_useless_spaces),
                ("normalize_repeating_chars", normalize_repeating_chars),
                ("normalize_repeating_words", normalize_repeating_words),
                ("strip", lambda x: x.strip()),
            ]
        )

    def fit_transform(self, series: pd.Series) -> Series:
        """Applies text preprocessing"""

        if self.lemmatizer_when == "first":
            with st.spinner("Lemmatizing"):
                series = self.lemmatizer(series)

        with st.spinner("Cleaning"):
            series = series.progress_map(self.cleaning_pipeline)

        if self.lemmatizer_when == "last":
            with st.spinner("Lemmatizing"):
                series = self.lemmatizer(series)

        return series
