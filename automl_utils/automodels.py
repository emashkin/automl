import os
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .feature_selector import GBMFeatureSelector
from .gbm_trainer import GBMTrainer
from .feature_utils import (BasicFeaturePreprocessor, FeatureGenerator,
                            TimeSeriesReverseOHE, define_task_type,
                            MeanTargetEncoderSmoothed, FeatureSpaceDefiner, ConstantColumnsRemover)
from .utils import get_dataset_size
from .memory_optimizers import MemUsageOptimizer


class AutoModel():
    '''
    Class that implements meta algorithm to solve machine learning problem on arbitrary dataset.

    It incorporates all the logic required to train ML meta model including:
        - automatic data cleaning,
        - automatic feature preprocessing,
        - automatic feature generation,
        - automatic model selection,
        - automatic hyperparameter tuning
        - etc.
    Class' interface follows sklearn fit/predict style.

    Args:
        init_time: timestamp of training process start.
            If not specified then class initialization timestamp is used.
        mode: trainning mode for the model.
            Could be either 'regression' or 'classification'. All other inputs are
            considered as 'classification' problem.
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self,
                 init_time: float = None,
                 mode: str = 'regression',
                 verbose: bool = False):
        self.mode = mode
        self.verbose = verbose
        self.init_time = init_time if init_time is not None else time.time()
        self.df_read_chunk_size_ = None
        self.OHE_MAX_VALUES = None
        self.task_type = {}
        self.not_negative_ts = False
        self.f_spaces = {}
        self.df_size = None
        self.columns_to_use = []

        self.models_history = {}
        self.best_model = None
        self.best_score = -np.inf

        self.mem_optimizer = None
        self.basic_preprocessor = None
        self.feature_generator = None
        self.constant_columns_remover_1 = None
        self.constant_columns_remover_2 = None
        self.mean_encoder = None
        self.reverse_ohe = None
        self.feature_selector = None


    def remove_corrupted_rows(self,
                              X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method removes corrupted row in a training dataset.

        Corrupted rows are:
            - Duplicates of rows,
            - Identical rows except target column,
            - Rows with Nan values in target columns

        Args:
            X: training dataset to clean

        Returns:
            DataFrame with corrupted rows dropped
        '''
        zero_time = time.time()
        in_rows = X.shape[0]

        # First drop completly identical rows with identical target
        columns_subset = [col for col in X.columns if col not in ['line_id']]
        X.drop_duplicates(subset=columns_subset, keep='first', inplace=True)

        # Then drop identical rows with different target as it brings up uncertainty
        columns_subset = [col for col in X.columns if col not in ['target', 'line_id']]
        X.drop_duplicates(subset=columns_subset, keep=False, inplace=True)

        # Drop rows where target is nan
        if 'target' in X.columns:
            X.dropna(subset=['target'], axis=0, inplace=True)
        out_rows = X.shape[0]
        if self.verbose:
            print('GC: {} corrupted rows dropped'.format(in_rows - out_rows))
            print('GC: completed in {:.2f} sec'.format(time.time() - zero_time))
        X.reset_index(drop=True, inplace=True)
        return X


    def compare_to_best_model(self, model: BaseEstimator, score: float):
        '''
        Method that compares score of model candidate with the best estimator score.

        Args:
            model: estimator cndidate to compare with best estimator
            score: score of estimator candidate

        Returns:
            None
        '''
        if score > self.best_score:
            self.best_model = model
            self.best_score = score


    def fit(self, X: pd.DataFrame):
        '''
        Implements meta model learning algorithm.

        It is supposed that trainin dataset contains only the following columns:
            line_id — an Id for each line
            target — target variable (only for train dataset), continuous variable for regression
                tasks and binary labels (0/1) for classification
            <type>_<feature number> — type of the feature (type):
                id — Id, special purpose categorical variable like User_id
                datetime — date feature in 2010-01-01 or 2010-01-01 10:10:10 format
                number — number feature (also could be continuous, categorical or binary variable)
                string — string feature

        Args:
            X: Train dataset. Its size isn't supposed to exceed 3 Gb.

        Returns:
            An instance of self
        '''
        start_time = time.time()
        self.df_read_chunk_size_ = max(X.shape[0], 100000)

        # Reduce RAM usage
        self.mem_optimizer = MemUsageOptimizer(verbose=self.verbose)
        X = self.mem_optimizer.fit_transform(X)

        # Remove corrupted rows
        X = self.remove_corrupted_rows(X)

        # Set max number of unique values in a column to consider it as categorical
        # but not exceeding 2% of samples in dataset
        # 20 < max ohe values = 2% of samples  < 100
        self.OHE_MAX_VALUES = max(min(X.shape[0]//50, 100), 20)

        # Define feature spaces
        space_definer = FeatureSpaceDefiner(ohe_max_values=self.OHE_MAX_VALUES)
        self.f_spaces = space_definer.define_feature_spaces(X)

        # Remove constant columns
        self.constant_columns_remover_1 = ConstantColumnsRemover(self.f_spaces,
                                                                 verbose=self.verbose)
        X = self.constant_columns_remover_1.fit_transform(X)

        # Define ML task type
        self.task_type = define_task_type(X, self.f_spaces, verbose=self.verbose)

        if self.task_type['is_timeseries']:
            # Sort Data by time
            zero_time = time.time()
            X.sort_values(by=['datetime_0'], inplace=True)
            if self.verbose:
                eval_time = time.time() - zero_time
                print('Data sorted for timeseries: completed in {:.2f} sec'.format(eval_time))

        # Perform basic preprocessing
        self.basic_preprocessor = BasicFeaturePreprocessor(mode=self.mode,
                                                           f_spaces=self.f_spaces,
                                                           task_type=self.task_type,
                                                           ohe_max_values=self.OHE_MAX_VALUES,
                                                           verbose=self.verbose)
        X = self.basic_preprocessor.fit_transform(X)

        # If it's a timeseries task, remove one hot encoded
        # time features to avoid duplication further
        self.df_size = get_dataset_size(X)
        if (self.mode == 'regression' and self.task_type['is_timeseries']
                and self.df_size in ['tiny', 'small']):
            self.reverse_ohe = TimeSeriesReverseOHE(self.f_spaces, verbose=self.verbose)
            X = self.reverse_ohe.fit_transform(X)

        # Generate new features
        self.feature_generator = FeatureGenerator(mode=self.mode,
                                                  f_spaces=self.f_spaces,
                                                  task_type=self.task_type,
                                                  create_feature_interactions=False,
                                                  verbose=self.verbose)
        X = self.feature_generator.fit_transform(X)

        # Remove constant features among newly generated features
        self.constant_columns_remover_2 = \
            ConstantColumnsRemover(self.f_spaces, verbose=self.verbose)
        X = self.constant_columns_remover_2.fit_transform(X)

        # Mean target encode string, categorical and id features
        # Use smoothed encoding schema
        self.mean_encoder = MeanTargetEncoderSmoothed(f_spaces=self.f_spaces,
                                                      task_type=self.task_type,
                                                      verbose=self.verbose)
        X = self.mean_encoder.transform_train(X)

        # Keep only useful columns defined in feature space
        self.columns_to_use = (self.f_spaces['binary'] +
                               self.f_spaces['categorical'] +
                               self.f_spaces['number'] +
                               self.f_spaces['id'])
        y = X['target']
        X.drop(X.columns.difference(self.columns_to_use), axis=1, inplace=True)

        # Select and keep only top usefull features
        self.feature_selector = GBMFeatureSelector(f_spaces=self.f_spaces,
                                                   task_type=self.task_type,
                                                   mode=self.mode,
                                                   gain_threshold=0.9999,
                                                   max_features=300,
                                                   create_feature_interactions=False,
                                                   n_iter=3,
                                                   verbose=self.verbose)
        X = self.feature_selector.fit_transform(X, y)

        if self.verbose:
            print('Data has been processed: X shape is {}. X occupies: {:.2f} Mb'.
                  format(X.shape, X.memory_usage(deep=True).sum() / 1024**2))

        # Fit GBM model
        gbm = GBMTrainer(mode=self.mode,
                         f_spaces=self.f_spaces,
                         task_type=self.task_type,
                         init_time=self.init_time,
                         verbose=self.verbose)
        gbm.fit(X, y)

        # Set this model as the best one
        self.best_model = gbm.best_estimator_
        return self


    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method makes predictions for test dataset using best estimator found during training.

        Args:
            X: test dataset to make predictions on

        Returns:
            Pandas DataFrame contains two columns:
                line_id: line number in dataset
                prediction: predicted target variable. For regression - absolute value of target
                    variable. For classification - predicted positive class probability.
        '''
        prediction_dframe = X[['line_id']]

        # Optimize RAM usage
        X = self.mem_optimizer.transform(X)

        X = self.constant_columns_remover_1.transform(X)

        # Perform basic preprocessing
        X = self.basic_preprocessor.transform(X)

        # If it's a timeseries task remove ohe-encoded time features
        if self.reverse_ohe is not None:
            X = self.reverse_ohe.transform(X)

        # Generate new features
        X = self.feature_generator.transform(X)

        # Remove constant features among newly generated features
        X = self.constant_columns_remover_2.transform(X)

        # Mean encode categorical, binary and id features
        X = self.mean_encoder.transform_test(X)

        # Keep only columns that were used during training
        X.drop(X.columns.difference(self.columns_to_use), axis=1, inplace=True)

        # Remove features not correlated to target
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)

        if self.mode == 'regression':
            prediction = self.best_model.predict(X)
            # If it's none negative timeseries task, remove
            # negative values in prediction to get some score inprovement
            if self.task_type['is_not_negative_timeseries']:
                prediction = np.maximum(0.0, prediction)
        else:
            prediction = self.best_model.predict_proba(X)[:, 1]

        prediction_dframe.loc[:, 'prediction'] = prediction
        return prediction_dframe
