from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from stqdm import stqdm

from .configs import ModelConfigs

stqdm.pandas()


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
