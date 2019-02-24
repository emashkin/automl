import time
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb


class GBMFeatureSelector():
    '''
    Selects best features based on LightGBM feature importances.

    Selection is performed several times with randomization:
        - different LGBM hyperparameters
        - different subsamples from train dataset

    All feature importances are being avereged among all iterations. Only top n features remain.

    Also feature selector tries to generate new feature interactions based on number of splits.
    To avoid feature explosion it is limited by interactions of top 4 split features.

    Args:
        f_spaces: dict with feature spaces description
        task_type: some machine learning task type properties to provide better learning strategy
        mode: trainning mode for the model. Could be either 'regression' or 'classification'.
            All other inputs are considered as 'classification' problem.
        max_features: maximum number of most informative features to keep
        gain_threshold: maximum value of cumulative information gain to reamin.
            E.g. if gain_threshold = 0.99 it keeps top features that give not more that 99% of
            total information gain.
        n_iter: number of randomized LGBM training iterations
        create_feature_interactions: If True, interactions of top informative featres will
            be created.
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self,
                 f_spaces: dict,
                 task_type: dict,
                 mode: str = 'regression',
                 max_features: int = 300,
                 gain_threshold: float = 0.9999,
                 n_iter: int = 5,
                 create_feature_interactions: bool = False,
                 verbose: bool = False):
        self.gain_threshold = gain_threshold
        self.create_feature_interactions = create_feature_interactions
        self.max_features = max_features
        self.n_iter = n_iter
        self.f_spaces = f_spaces
        self.task_type = task_type
        self.mode = mode
        self.columns_to_keep = []
        self.columns_to_drop = []
        self.feature_gain = None
        self.feature_split = None
        self.columns_to_pair = []
        self.new_cat_column_value_encoder = {}
        self.verbose = verbose
        self.feature_importances = []
        self.failed_to_select_features = False


    def fit(self,
            X: pd.DataFrame,
            y: np.ndarray):
        '''
        Performs several randomized feature selection iterations and remember feature list to keep.

        Args:
            X: datset to select most informative features
            y: target vector

        Returns:
            An instance of self
        '''
        zero_time = time.time()

        params = {
            'objective': 'regression' if self.mode == 'regression' else 'binary',
            'metric': 'rmse' if self.mode == 'regression' else 'auc',
            'n_estimators': 100,
            'verbosity': -1,
            'seed': 13
        }
        basic_params = {
            'objective': 'regression' if self.mode == 'regression' else 'binary',
            'metric': 'rmse' if self.mode == 'regression' else 'auc',
            'num_leaves': 16,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_weight': 5,
            'verbosity': -1
        }
        self.feature_importances = []
        self.feature_gain = pd.Series(np.zeros_like(X.columns))
        self.feature_split = pd.Series(np.zeros_like(X.columns))
        self.new_cat_column_value_encoder = {}

        train_data = lgb.Dataset(X, label=y, free_raw_data=False)

        for i in range(self.n_iter):
            inner_base_time = time.time()

            # Change hyperparameters
            params['seed'] = i**2 + i
            basic_params['num_leaves'] += i * 16
            basic_params['reg_alpha'] += 0.5 * i
            basic_params['reg_lambda'] += 0.5 * i

            estimator = lgb.train({**params, **basic_params}, train_set=train_data)

            self.feature_gain += pd.Series(estimator.feature_importance(importance_type='gain'))
            self.feature_split += pd.Series(estimator.feature_importance(importance_type='split'))

            if self.verbose:
                print('GBM FEATURE SELECTOR: Iteration {}/{} completed in {:.2f} sec'.format(\
                      i + 1, self.n_iter, time.time() - inner_base_time))

        # Average results
        self.feature_gain /= self.n_iter
        self.feature_split /= self.n_iter
        
        if self.feature_gain.sum() == 0:
            self.failed_to_select_features = True
            if self.verbose:
                print('GBM FEATURE SELECTOR: Failed to select features' +
                      '- no meaningful features found')
            return self

        self.feature_gain /= self.feature_gain.sum()
        self.feature_split /= self.feature_split.sum()

        self.feature_gain.sort_values(ascending=False, inplace=True)
        self.feature_split.sort_values(ascending=False, inplace=True)

        self.feature_gain = np.cumsum(self.feature_gain)
        self.feature_split = np.cumsum(self.feature_split)

        col_idx_keep = self.feature_gain[self.feature_gain <= self.gain_threshold].index
        col_idx_keep = col_idx_keep[:self.max_features]

        self.columns_to_keep = X.columns[col_idx_keep].tolist()
        self.columns_to_pair = X.columns[self.feature_split[:4].index]

        X_len_before_new_features = len(self.columns_to_keep)

        # Create feature interactions if required
        if self.create_feature_interactions:
            for col1, col2 in combinations(self.columns_to_pair, 2):
                # If numerical-numerical feature interaction
                if col1 in self.f_spaces['number'] and col2 in self.f_spaces['number']:
                    self.columns_to_keep += ['{}_{}_div'.format(col1, col2),
                                             '{}_{}_diff'.format(col1, col2),
                                             '{}_{}_mult'.format(col1, col2),
                                             '{}_{}_sum'.format(col1, col2)]
                    self.f_spaces['number'] += ['{}_{}_div'.format(col1, col2),
                                                '{}_{}_diff'.format(col1, col2),
                                                '{}_{}_mult'.format(col1, col2),
                                                '{}_{}_sum'.format(col1, col2)]
                # If categorical-binary feature interaction
                elif col1 in self.f_spaces['categorical'] + self.f_spaces['binary'] and \
                     col2 in self.f_spaces['categorical'] + self.f_spaces['binary']:

                    # Add new column to 'white' list and update feature space
                    self.columns_to_keep += ['{}_{}_interaction'.format(col1, col2)]
                    self.f_spaces['categorical'] += ['{}_{}_interaction'.format(col1, col2)]

                    # Remember encoding mapping dict for new column
                    new_col = X[col1].map(str) + X[col2].map(str)
                    new_col_unique = list((new_col).value_counts(dropna=False).index)
                    self.new_cat_column_value_encoder[col1, col2] = \
                        dict(zip(new_col_unique, range(len(new_col_unique))))
                else:
                    self.columns_to_keep += ['{}_{}_interaction'.format(col1, col2)]
                    self.f_spaces['number'] += ['{}_{}_interaction'.format(col1, col2)]

        self.columns_to_drop = [col for col in X.columns if col not in self.columns_to_keep]

        if self.verbose:
            eval_time = time.time() - zero_time
            n_created = len(self.columns_to_keep) - X_len_before_new_features
            print('GBM FEATURE SELECTOR: {} features selected, {} dropped, {} created'.format(\
                  X_len_before_new_features, len(self.columns_to_drop), n_created))
            print('GBM FEATURE SELECTOR: fit completed in {:.2f} sec'.format(eval_time))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Remains only top informative features and create new feature interactions if required.

        Args:
            X: datset to remove non informative features
            y: target vector

        Returns:
            A dataset of most informative features
        '''
        zero_time = time.time()

        if self.failed_to_select_features:
            if self.verbose:
                eval_time = time.time() - zero_time
                print('GBM FEATURE SELECTOR: No transform - ' +
                      'completed in {:.2f} sec'.format(eval_time))
            return X

        # Create feature interactions if required
        if self.create_feature_interactions:
            for col1, col2 in combinations(self.columns_to_pair, 2):
                # If numerical-numerical feature interaction
                if col1 in self.f_spaces['number'] and col2 in self.f_spaces['number']:
                    X['{}_{}_div'.format(col1, col2)] = X[col1] / (X[col2] + 1e-4)
                    X['{}_{}_div'.format(col1, col2)] = X[col2] / (X[col1] + 1e-4)
                    X['{}_{}_diff'.format(col1, col2)] = X[col1] - X[col2]
                    X['{}_{}_mult'.format(col1, col2)] = X[col1] * X[col2]
                    X['{}_{}_sum'.format(col1, col2)] = X[col1] + X[col2]

                # If categorical-binary feature interaction
                elif col1 in self.f_spaces['categorical'] + self.f_spaces['binary'] and \
                     col2 in self.f_spaces['categorical'] + self.f_spaces['binary']:
                    new_col = X[col1].map(str) + X[col2].map(str)
                    X['{}_{}_interaction'] = \
                        new_col.map(self.new_cat_column_value_encoder[col1, col2])
                    X['{}_{}_interaction'] = X['{}_{}_interaction'].fillna(-1).astype(np.int32)
                else:
                    X['{}_{}_interaction'] = X[col1] * X[col2]

        # Remain only most informative features
        X.drop(self.columns_to_drop, axis=1, inplace=True)

        if self.verbose:
            eval_time = time.time() - zero_time
            print('GBM FEATURE SELECTOR: transform completed in {:.2f} sec'.format(eval_time))
        return X


    def fit_transform(self, X, y):
        '''
        Consequently calls fit and predict methods
        '''
        self.fit(X, y)
        return self.transform(X)
