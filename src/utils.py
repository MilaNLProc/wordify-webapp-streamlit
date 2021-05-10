import base64
import re
from collections import OrderedDict
from typing import Callable, Dict, List

import altair as alt
import numpy as np
import pandas as pd
import spacy
import streamlit as st
from pandas.core.series import Series
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from stqdm import stqdm
from textacy.preprocessing import make_pipeline, normalize, remove, replace

from .configs import Languages, ModelConfigs, SupportedFiles

stqdm.pandas()


@st.cache
def get_logo(path):
    return Image.open(path)


# @st.cache(suppress_st_warning=True)
def read_file(uploaded_file) -> pd.DataFrame:

    file_type = uploaded_file.name.split(".")[-1]
    if file_type in set(i.name for i in SupportedFiles):
        read_f = SupportedFiles[file_type].value[0]
        return read_f(uploaded_file, dtype=str)

    else:
        st.error("File type not supported")


def download_button(dataframe: pd.DataFrame, name: str):
    csv = dataframe.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download</a>'
    st.write(href, unsafe_allow_html=True)


def encode(text: pd.Series, labels: pd.Series):
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


def wordifier(X, y, X_names: List[str], y_names: List[str], configs=ModelConfigs):

    n_instances, n_features = X.shape
    n_classes = len(y_names)

    # NOTE: the * 10 / 10 trick is to have "nice" round-ups
    sample_fraction = np.ceil((n_features / n_instances) * 10) / 10

    sample_size = min(
        # this is the maximum supported
        configs.MAX_SELECTION.value,
        # at minimum you want MIN_SELECTION but in general you want
        # n_instances * sample_fraction
        max(configs.MIN_SELECTION.value, int(n_instances * sample_fraction)),
        # however if previous one is bigger the the available instances take
        # the number of available instances
        n_instances,
    )

    # TODO: might want to try out something to subsample features at each iteration

    # initialize coefficient matrices
    pos_scores = np.zeros((n_classes, n_features), dtype=int)
    neg_scores = np.zeros((n_classes, n_features), dtype=int)

    with st.spinner("Wordifying!"):

        for _ in stqdm(range(configs.NUM_ITERS.value)):

            # run randomized regression
            clf = LogisticRegression(
                penalty="l1",
                C=configs.PENALTIES.value[np.random.randint(len(configs.PENALTIES.value))],
                solver="liblinear",
                multi_class="auto",
                max_iter=500,
                class_weight="balanced",
            )

            # sample indices to subsample matrix
            selection = resample(np.arange(n_instances), replace=True, stratify=y, n_samples=sample_size)

            # fit
            try:
                clf.fit(X[selection], y[selection])
            except ValueError:
                continue

            # record coefficients
            if n_classes == 2:
                pos_scores[1] = pos_scores[1] + (clf.coef_ > 0.0)
                neg_scores[1] = neg_scores[1] + (clf.coef_ < 0.0)
                pos_scores[0] = pos_scores[0] + (clf.coef_ < 0.0)
                neg_scores[0] = neg_scores[0] + (clf.coef_ > 0.0)
            else:
                pos_scores += clf.coef_ > 0
                neg_scores += clf.coef_ < 0

        # normalize
        pos_scores = pos_scores / configs.NUM_ITERS.value
        neg_scores = neg_scores / configs.NUM_ITERS.value

        # get only active features
        pos_positions = np.where(pos_scores >= configs.SELECTION_THRESHOLD.value, pos_scores, 0)
        neg_positions = np.where(neg_scores >= configs.SELECTION_THRESHOLD.value, neg_scores, 0)

        # prepare DataFrame
        pos = [(X_names[i], pos_scores[c, i], y_names[c]) for c, i in zip(*pos_positions.nonzero())]
        neg = [(X_names[i], neg_scores[c, i], y_names[c]) for c, i in zip(*neg_positions.nonzero())]

    posdf = pd.DataFrame(pos, columns="word score label".split()).sort_values(["label", "score"], ascending=False)
    negdf = pd.DataFrame(neg, columns="word score label".split()).sort_values(["label", "score"], ascending=False)

    return posdf, negdf


# more [here](https://github.com/fastai/fastai/blob/master/fastai/text/core.py#L42)
# and [here](https://textacy.readthedocs.io/en/latest/api_reference/preprocessing.html)
_re_space = re.compile(" {2,}")


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


class TextPreprocessor:
    def __init__(
        self, language: str, cleaning_steps: List[str], lemmatizer_when: str = "last", remove_stop: bool = True
    ) -> None:
        # prepare lemmatizer
        self.language = language
        self.nlp = spacy.load(Languages[language].value, exclude=["parser", "ner", "pos", "tok2vec"])
        self.lemmatizer_when = self._lemmatization_options().get(lemmatizer_when, None)
        self.remove_stop = remove_stop
        self._lemmatize = self._get_lemmatizer()

        # prepare cleaning
        self.cleaning_steps = [
            self._cleaning_options()[step] for step in cleaning_steps if step in self._cleaning_options()
        ]
        self.cleaning_pipeline = make_pipeline(*self.cleaning_steps) if self.cleaning_steps else lambda x: x

    def _get_lemmatizer(self) -> Callable:
        """Return the correct spacy Doc-level lemmatizer"""
        if self.remove_stop:

            def lemmatizer(doc: spacy.tokens.doc.Doc) -> str:
                """Lemmatizes spacy Doc and removes stopwords"""
                return " ".join([t.lemma_ for t in doc if t.lemma_ != "-PRON-" and not t.is_stop])

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
                ("remove_accents", remove.accents),
                ("remove_brackets", remove.brackets),
                ("remove_html_tags", remove.html_tags),
                ("remove_punctuation", remove.punctuation),
                ("replace_currency_symbols", replace.currency_symbols),
                ("replace_emails", replace.emails),
                ("replace_emojis", replace.emojis),
                ("replace_hashtags", replace.hashtags),
                ("replace_numbers", replace.numbers),
                ("replace_phone_numbers", replace.phone_numbers),
                ("replace_urls", replace.urls),
                ("replace_user_handles", replace.user_handles),
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


def plot_labels_prop(data: pd.DataFrame, label_column: str):

    source = data["label"].value_counts().reset_index().rename(columns={"index": "Labels", label_column: "Counts"})

    source["Proportions"] = ((source["Counts"] / source["Counts"].sum()).round(3) * 100).map("{:,.2f}".format) + "%"

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x="Labels:O",
            y="Counts:Q",
        )
    )

    text = bars.mark_text(align="center", baseline="middle", dy=15).encode(text="Proportions:O")

    return (bars + text).properties(height=300)


def plot_nchars(data: pd.DataFrame, text_column: str):
    source = data[text_column].str.len().to_frame()

    plot = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            alt.X(f"{text_column}:Q", bin=True, axis=alt.Axis(title="# chars per text")),
            alt.Y("count()", axis=alt.Axis(title="")),
        )
    )

    return plot.properties(height=300)


def plot_score(data: pd.DataFrame, label_col: str, label: str):

    source = data.loc[data[label_col] == label].sort_values("score", ascending=False).head(100)

    plot = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            y=alt.Y("word:O", sort="-x"),
            x="score:Q",
        )
    )

    return plot.properties(height=max(30 * source.shape[0], 50))
