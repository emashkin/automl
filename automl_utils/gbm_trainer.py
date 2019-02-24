import time
import gc
import os
import numpy as np
import pandas as pd

from hyperopt import fmin, hp, tpe, STATUS_OK, space_eval, Trials
import lightgbm as lgb

from .utils import data_split, subsample_data


TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
GLOBAL_TIME_FACTOR = 1   # 1.35


class GBMTrainer():
    '''
    Trains LightGBM estimator.

    Training contains three main stages:
        Stage 1: Find optimal learning rate for given data
        Find optimal learning rate for hyperparameter tuning stage having fixed number
        of estimators (300). Performs 10 hyperopt iterations to find optimal learning rate.
        This stage is also required to evaluate average training time for one iteration and
        estimate time required to perform hyperparameter tuning and final model training

        Stage 2: Hyper parameter tuning
        Use hyperopt to find other optimal parameters of LGBM. Number of iterations evalutaed
        based on remaining available time for training and average training time per iteration.

        Stage 3: Train final model using found parameters increasing n_estimators and decresing
        learning rate by same factor of 6 (thumb rule).

    Vlidation strategies:
        For datasets under 1000 samples performs 10-folds cross-validation.
        For datasets more than 1000 but less than 20 000 samples preforms 4-folds cross-validation.
        For large datasets above 20 000 samples performs 50% holdout validation strategy.

    Args:
        f_spaces: dict with feature spaces description
        task_type: some machine learning task type properties to provide better learning strategy
        init_time: timestamp of global training start. Needed to evaluate remaining time.
        mode: trainning mode for the model. Could be either 'regression' or 'classification'.
            All other inputs are considered as 'classification' problem.
        holdout_ratio: size of holdout validation dataset for final model training
        rounds_per_iteration: int = 500,
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self,
                 f_spaces: dict,
                 task_type: dict,
                 init_time: float,
                 mode: str = 'regression',
                 holdout_ratio: float = 0.3,
                 verbose: bool = False):
        self.mode = mode
        self.f_spaces = f_spaces
        self.estimator_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.verbose = verbose
        self.task_type = task_type
        self.init_time = init_time
        self.holdout_ratio = holdout_ratio


    def time_left(self):
        '''
        Evaluates remaining time availabel for training
        '''
        return TIME_LIMIT - (time.time() - self.init_time)


    def fit(self, X, y):
        '''
        Trains LightGBM estimator using meta algorithm

        Args:
            X: train dataset
            y: train target vector

        Returns:
            An instane of self
        '''
        params = {
            'objective': 'regression' if self.mode == 'regression' else 'binary',
            'metric': 'rmse' if self.mode == 'regression' else 'auc',
            'n_estimators': 300,
            'verbosity': -1,
            'seed': 13
        }

        # Stage 1 and Stage 2
        # Find optimal hyperparameters on a subset of data
        if X.shape[0] < 20000:
            n_folds = 10 if X.shape[0] < 1000 else 4
            X_sample, y_sample = subsample_data(X, y, subsample_ratio=0.5, mode=self.mode)
            hyperparams = self.hyperopt_lightgbm_cv(X_sample, y_sample, params=params,
                                                    mode=self.mode, n_folds=n_folds)
        else:
            X_sample, y_sample = subsample_data(X, y, subsample_ratio=0.35, mode=self.mode)
            hyperparams = self.hyperopt_lightgbm_holdout(X_sample, y_sample, params=params,
                                                         holdout_fraction=0.4, mode=self.mode)

        # Free memory to avoid crash
        del(X_sample, y_sample)
        gc.collect()

        # Stage 3
        # Split data into train and validation
        X_train, X_val, y_train, y_val = \
            data_split(X, y, test_size=self.holdout_ratio, mode=self.mode)
        del(X, y)
        gc.collect()

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

        if self.verbose:
            time_passed = time.time() - self.init_time
            print('GBM TRAINER: Time passed since begining: {:.2f}'.format(time_passed))
            print('GBM TRAINER: Available final model train time: {:.2f}'.format(self.time_left()))

        # Decrease learning rate and increse number of trees
        hyperparams['learning_rate'] = params['learning_rate'] / 8
        params['learning_rate'] /= 8
        params['n_estimators'] *= 10

        # Train final model
        zero_time = time.time()
        self.estimator_ = lgb.train({**params, **hyperparams},
                                    train_set=train_data,
                                    valid_sets=valid_data,
                                    # num_boost_round=3000,
                                    early_stopping_rounds=200,
                                    verbose_eval=100)
        print('GBM TRAINER: Actual final model train time: {:.2f}'.format(time.time() - zero_time))

        self.best_estimator_ = self
        self.best_score_ = 1.0
        return self


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Makes predictions for a given dataset.

        Args:
            X: test dataframe

        Returns:
            Predicted target vector:
                For regression task returns absolute predicted values of target variable
                For classification returns class label for each sample
        '''
        return self.estimator_.predict(X)


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Makes probability predictions for a given dataset.

        Args:
            X: test dataframe

        Returns:
            Vector of predicted probabilities for each class
        '''
        prediction = self.estimator_.predict(X)
        # Make the output of the model compatible with sklearn API
        # Predict probabilities for 0 and 1 classes
        prediction = np.vstack([1 - prediction, prediction]).T
        return prediction


    def hyperopt_lightgbm_cv(self,
                             X: pd.DataFrame,
                             y: np.ndarray,
                             params: dict,
                             mode: str,
                             n_folds: int = 3) -> dict:
        '''
        Finds optimal learning rate for n_estimators=300 and performs hyperparameter tuning.

        Args:
            X: dataset to performe hyperparameter tuning on
            y: target vector
            params: LGBM base parameters
            mode: trainning mode for the model. Could be either 'regression' or 'classification'.
                All other inputs are considered as 'classification' problem.
            n_folds: number of folds in cross-validation

        Returns:
            Dict of best LGBM hyperparameters found
        '''
        print('GBM TRAINER: {}-Fold CV Hyperparameters tuning on {}.'.format(n_folds, X.shape))

        train_data = lgb.Dataset(X, label=y)
        is_stratified_cv = params['metric'] == 'auc'


        def objective(hyperparams: dict):
            '''
            Objective function to optimize via hyperopt.

            Takes LGBM hyperparameters and returns cross-validation score.

            Args:
                hyperparams: LGBM hyperparameters to validate

            Returns:
                Dict of validation score and evaluation status
            '''
            eval_hist = lgb.cv({**params, **hyperparams},
                               stratified=is_stratified_cv,
                               train_set=train_data,
                               nfold=n_folds,
                               shuffle=False,
                               num_boost_round=500,
                               early_stopping_rounds=100,
                               verbose_eval=-1)
            if params['metric'] == 'auc':
                score = eval_hist['auc-mean'][-1]
            else:
                score = eval_hist['rmse-mean'][-1]

            if not mode == 'regression':
                score = -score
            return {'loss': score, 'status': STATUS_OK}

        # Find optimal lr having n_estimators = 300
        # Define hyperparameters space
        lr_space = {
            'learning_rate': hp.uniform('learning_rate', 0.03, 0.15)
        }

        # Find optimal learning rate
        lr_search_zero_time = time.time()
        trials = Trials()
        best_lr = fmin(fn=objective, space=lr_space, trials=trials, algo=tpe.suggest,
                       max_evals=10, verbose=-1, rstate=np.random.RandomState(13))
        lr_hyperparams = space_eval(lr_space, best_lr)

        # Evaluate average time per iteration
        time_per_tuning_iteration = ((time.time() - lr_search_zero_time) / 10) * 1.5
        est_final_model_train_time = (time_per_tuning_iteration * 8 * 1.5 / n_folds)

        if self.verbose:
            print('GBM TRAINER: Available time {:.2f} sec'.format(self.time_left()))

        # Evaluate number of iterations till time is up
        global_mult = GLOBAL_TIME_FACTOR
        N = 1
        time_left = self.time_left()

        while time_left - N * time_per_tuning_iteration * global_mult > \
            est_final_model_train_time + TIME_LIMIT * 0.1:
            N += 1
            time_left = self.time_left()

        est_hp_search_time = time_per_tuning_iteration * N

        print('GBM TRAINER: Number of hyperparameter search iterations: {}'.format(N))
        print('GBM TRAINER: Estimated hyperparameters search' +
              'time {:.2f} sec'.format(est_hp_search_time))
        print('GBM TRAINER: Estimated final model train ' +
              'time {:.2f} sec'.format(est_final_model_train_time))

        # Perform Hyperparameter tuning
        zero_search_time = time.time()
        params['learning_rate'] = lr_hyperparams['learning_rate']
        space = {
            'max_depth': hp.choice('max_depth', [-1, 2, 3, 4, 5, 6, 8]),
            'num_leaves': hp.choice('num_leaves', np.linspace(3, 200, 50, dtype=int)),
            'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
            'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
            'bagging_freq': hp.choice('bagging_freq', np.linspace(0, 50, 10, dtype=int)),
            'reg_alpha': hp.uniform('reg_alpha', 0, 32),
            'reg_lambda': hp.uniform('reg_lambda', 0, 32),
            'min_child_weight': hp.uniform('min_child_weight', 0.5, 12)
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=N,
                    verbose=1, rstate=np.random.RandomState(13))

        hyperparams = space_eval(space, best)

        eval_time = time.time() - zero_search_time
        print('GBM TRAINER: Actual hyperparameters search time: {:.2f}'.format(eval_time))
        print('Best cv params:', hyperparams)

        return hyperparams


    def hyperopt_lightgbm_holdout(self,
                                  X: pd.DataFrame,
                                  y: np.ndarray,
                                  params: dict,
                                  mode: str,
                                  holdout_fraction: int = 0.5) -> dict:
        '''
        Finds optimal learning rate for n_estimators=300 and performs hyperparameter tuning.

        Args:
            X: dataset to performe hyperparameter tuning on
            y: target vector
            params: LGBM base parameters
            mode: trainning mode for the model. Could be either 'regression' or 'classification'.
                All other inputs are considered as 'classification' problem.
            holdout_fraction: size of holdout validations dataset

        Returns:
            Dict of best LGBM hyperparameters found
        '''

        X_train, X_valid, y_train, y_valid = \
            data_split(X, y, test_size=holdout_fraction, mode=mode)

        print('GBM TRAINER: {:.0f}/{:.0f}% holdout hyperparameters tuning on {}.'.format(\
            (1 - holdout_fraction)*100, holdout_fraction*100, X_train.shape))

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)


        def objective(hyperparams):
            '''
            Objective function to optimize via hyperopt.

            Takes LGBM hyperparameters and returns holdout validation score.

            Args:
                hyperparams: LGBM hyperparameters to validate

            Returns:
                Dict of validation score and evaluation status
            '''
            model = lgb.train({**params, **hyperparams},
                              train_set=train_data,
                              valid_sets=valid_data,
                              early_stopping_rounds=100,
                              verbose_eval=-1 if self.verbose else -1)

            score = model.best_score['valid_0'][params['metric']]
            if mode != 'regression':
                score = -score
            return {'loss': score, 'status': STATUS_OK, 'best_iter': model.best_iteration}

        # Find optimal lr having n_estimators = 300
        lr_space = {
            'learning_rate': hp.uniform('learning_rate', 0.02, 0.18)
        }

        # Find optimal learning rate
        lr_search_zero_time = time.time()
        trials = Trials()
        best_lr = fmin(fn=objective, space=lr_space, trials=trials, algo=tpe.suggest,
                       max_evals=10, verbose=-1, rstate=np.random.RandomState(13))
        lr_hyperparams = space_eval(lr_space, best_lr)

        # Evaluate average time per iteration
        time_per_tuning_iteration = ((time.time() - lr_search_zero_time) / 10) * 1.25
        est_final_model_train_time = (time_per_tuning_iteration * 10 * 1.5)
        print('GBM TRAINER: Available time {:.2f} sec'.format(self.time_left()))

        # Evaluate number of iterations till time is up
        global_mult = GLOBAL_TIME_FACTOR
        N = 1
        time_left = self.time_left()
        while time_left - N * time_per_tuning_iteration * global_mult > \
            est_final_model_train_time + TIME_LIMIT * 0.1:
            N += 1
            time_left = self.time_left()

        est_hp_search_time = time_per_tuning_iteration * N

        print('GBM TRAINER: Number of hyperparameter search iterations: {}'.format(N))
        print('GBM TRAINER: Estimated hyperparameters search' +
              'time {:.2f} sec'.format(est_hp_search_time))
        print('GBM TRAINER: Estimated final model train ' +
              'time {:.2f} sec'.format(est_final_model_train_time))

        zero_search_time = time.time()
        params['learning_rate'] = lr_hyperparams['learning_rate']

        # Perform Hyperparameter tuning
        space = {
            'max_depth': hp.choice('max_depth', [-1, 2, 3, 4, 5, 6, 8, 10, 12, 16]),
            'num_leaves': hp.choice('num_leaves', np.linspace(3, 200, 50, dtype=int)),
            'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
            'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
            'bagging_freq': hp.choice('bagging_freq', np.linspace(0, 20, 10, dtype=int)),
            'reg_alpha': hp.uniform('reg_alpha', 0, 32),
            'reg_lambda': hp.uniform('reg_lambda', 0, 32),
            'min_child_weight': hp.uniform('min_child_weight', 0.5, 12)
        }

        trials = Trials()
        best = fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest,
                    max_evals=N, verbose=1, rstate=np.random.RandomState(13))
        hyperparams = space_eval(space, best)

        eval_time = time.time() - zero_search_time
        print('GBM TRAINER: Actual hyperparameters search time: {:.2f}'.format(eval_time))
        return hyperparams
