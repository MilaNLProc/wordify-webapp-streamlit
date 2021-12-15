from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from .configs import InputTransformConfigs, ModelConfigs


def input_transform(
    text: pd.Series, labels: pd.Series, configs=InputTransformConfigs
) -> Dict[str, np.ndarray]:
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
        ngram_range=configs.NGRAM_RANGE.value,  # maximum 3-ngrams
        min_df=configs.MIN_DF.value,
        max_df=configs.MAX_DF.value,
        sublinear_tf=configs.SUBLINEAR.value,
    )
    label_encoder = LabelEncoder()

    X = tfidf_vectorizer.fit_transform(text.values)
    y = label_encoder.fit_transform(labels.values)

    return {
        "X": X,
        "y": y,
        "X_names": np.array(tfidf_vectorizer.get_feature_names_out()),
        "y_names": label_encoder.classes_,
    }


def wordifier(
    X: np.ndarray,
    y: np.ndarray,
    X_names: List[str],
    y_names: List[str],
    configs=ModelConfigs,
) -> List[Tuple[str, float, str]]:

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

    pbar = st.progress(0)
    for i, _ in enumerate(range(configs.NUM_ITERS.value)):

        # run randomized regression
        clf = LogisticRegression(
            penalty="l1",
            C=configs.PENALTIES.value[np.random.randint(len(configs.PENALTIES.value))],
            solver="liblinear",
            multi_class="auto",
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )

        # sample indices to subsample matrix
        selection = resample(
            np.arange(n_instances), replace=True, stratify=y, n_samples=sample_size
        )

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

        pbar.progress(round(i / configs.NUM_ITERS.value, 1))

    # normalize
    pos_scores = pos_scores / configs.NUM_ITERS.value
    neg_scores = neg_scores / configs.NUM_ITERS.value

    # get only active features
    pos_positions = np.where(
        pos_scores >= configs.SELECTION_THRESHOLD.value, pos_scores, 0
    )
    neg_positions = np.where(
        neg_scores >= configs.SELECTION_THRESHOLD.value, neg_scores, 0
    )

    # prepare DataFrame
    pos = [
        (X_names[i], pos_scores[c, i], y_names[c])
        for c, i in zip(*pos_positions.nonzero())
    ]
    neg = [
        (X_names[i], neg_scores[c, i], y_names[c])
        for c, i in zip(*neg_positions.nonzero())
    ]

    return pos, neg


def output_transform(
    pos: List[Tuple[str, float, str]], neg: List[Tuple[str, float, str]]
) -> DataFrame:
    posdf = pd.DataFrame(pos, columns="word score label".split()).sort_values(
        ["label", "score"], ascending=False
    )
    posdf["correlation"] = "positive"
    negdf = pd.DataFrame(neg, columns="word score label".split()).sort_values(
        ["label", "score"], ascending=False
    )
    negdf["correlation"] = "negative"

    output = pd.concat([posdf, negdf], ignore_index=False, axis=0)
    output.columns = output.columns.str.title()

    return output
