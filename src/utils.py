from sklearn.preprocessing import LabelEncoder, normalize
from typing import List
import numpy as np
import logging


def log_results(filename, log_info):
    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(message)s',
    )
    logging.info(log_info)


def split_features_and_classes(
    df,
    class_col: str,
    encode: List[str] = None,
    drop: List[str] = None
):
    if encode is not None:
        for e in encode:
            df[e] = LabelEncoder().fit_transform(df[e])
    if drop is not None:
        df = df.drop(columns=drop) # all 'oral' entries are true, so useless.
    x = np.asarray(df.drop(columns=[class_col])) #TODO: Change to `class_col`
    x = normalize(x)
    y = np.asarray(df[class_col])
    return df, x, y


def shuffle_and_keep_for_validation(x, y, cv):
    nsamples = len(x)
    n_cv = int(cv * nsamples)
    n_train = nsamples - n_cv
    perm = np.random.permutation(nsamples)
    x, y = x[perm], y[perm]
    x_train, y_train = x[:n_train], y[:n_train]
    x_cv, y_cv = x[n_train:nsamples], y[n_train:nsamples]
    return x_train, y_train, x_cv, y_cv