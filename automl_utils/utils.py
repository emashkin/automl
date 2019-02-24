import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset_size(X: pd.DataFrame) -> str:
    '''
    Returns dataset type category based on its size.

    Args:
        X: dataset

    Returns:
        Dataset size type
    '''
    df_size = X.memory_usage(deep=True).sum() / 1024**2
    if df_size <= 8:
        return 'tiny'
    if df_size <= 32:
        return 'small'
    if df_size <= 256:
        return 'medium'
    if df_size <= 1024:
        return 'large'
    if df_size <= 1536:
        return 'xlarge'
    return 'giant'


def data_split(X, y, test_size=0.2, mode='regression'):
    '''
    Splits data into train and test parts.

    Args:
        X: Dataset to split
        y: target vector to split

    Returns:
        tuple of X_train, X_validation, y_train, y_validation
    '''
    if mode == 'regression':
        X_train, X_val, y_train, y_val = \
            train_test_split(X, y, test_size=test_size, random_state=13)
    else:
        X_train, X_val, y_train, y_val = \
            train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y, random_state=13)
    return X_train, X_val, y_train, y_val


def subsample_data(X: pd.DataFrame,
                   y: np.ndarray,
                   mode: str,
                   subsample_ratio: float = None,
                   subsample_mb_size: float = 4.0) -> tuple:
    '''
    Susamples data from original dataset.

    Args:
        X: Dataset to split
        y: target vector to split
        subsample_ratio: size of subsampled data relatively original data
        subsample_mb_size: if subsample_ratio is not specified, subsampled data
            will not exceed subsample_mb_size

    Returns:
        tuple of subsampled X and y of required length
    '''
    df_size = X.memory_usage(deep=True).sum() / 1024**2

    if df_size <= subsample_mb_size:
        return X, y

    if subsample_ratio is None:
        mb_per_row = df_size / X.shape[0]
        rows_to_select = subsample_mb_size // mb_per_row
        subsample_ratio = rows_to_select / X.shape[0]
    else:
        rows_to_select = int(X.shape[0] * subsample_ratio)

    if mode == 'regression':
        X_sample = X.sample(rows_to_select, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=subsample_ratio,
                                                    random_state=13, shuffle=True, stratify=y)
    return X_sample, y_sample
