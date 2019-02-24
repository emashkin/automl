import time
import datetime
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from .ru_holidays_calendar import ru_holidays


def define_task_type(X: pd.DataFrame,
                     f_spaces: dict,
                     verbose: bool = False) -> dict:
    '''
    Defines some machine learning task type properties to provide better learning strategy.

    Task types could be:
        - timeseries task
        - timeseries task with id variable, e.g. timeseries for several user_ids
        - not negative time series (negative predictions could be cropped)
        - other

    Args:
        X: train dataset
        f_spaces: dict with feature spaces description
        verbose: boolean, default False
            If True some information during processing will be output

    Returns:
        Dict with found dataset types: {
            'is_timeseries': True/False,
            'with_id': True/False,
            'is_not_negative_timeseries': True/False
        }
    '''
    zero_time = time.time()
    task_type = {
        'is_timeseries': False,
        'with_id': False,
        'is_not_negative_timeseries': False
    }

    if len(f_spaces['datetime']) == 1:
        unique_dates = X['datetime_0'].nunique(dropna=False)
        if unique_dates > X.shape[0] * 0.98:
            task_type['is_timeseries'] = True
            if (X['target'] >= 0).all():
                task_type['is_not_negative_timeseries'] = True

        if len(f_spaces['id']) == 1:
            unique_ids = X['id_0'].nunique(dropna=False)
            if ((unique_ids * unique_dates > X.shape[0] * 0.98)
                    and (unique_ids * unique_dates <= X.shape[0] * 1.05)):

                task_type['is_timeseries'] = True
                task_type['with_id'] = True
                if (X['target'] >= 0).all():
                    task_type['is_not_negative_timeseries'] = True
    if verbose:
        print('TASK TYPE: {}'.format(task_type))
        print('TASK TYPE: completed in {:.2f} sec'.format(time.time() - zero_time))
    return task_type


class FeatureSpaceDefiner():
    '''
    Class that defines feature spaces(e.g. binary features, categorical features etc.)
    in a dataset for the further processing.

    Args:
        max_categorical_values: int, default 10
            Maximum number of unique values in a column to consider as categorical
        id_columns: list, default empty list
            List of id column names that have special nature and you'd like
            to handle them in a special way. Requires some prior knowledge
            about such columns so should be specified manually.
        verbose : boolean, default False
            If True some information during processing will be output
    '''
    def __init__(self,
                 ohe_max_values: int = 10,
                 verbose: bool = False):
        self.verbose = verbose
        self.OHE_MAX_VALUES = ohe_max_values
        self.f_spaces = {}


    def define_feature_spaces(self, X: pd.DataFrame) -> dict:
        '''
        Defines featurespaces in given dataset

        Args:
            X: dataframe to define feature spaces in.

        Returns:
            Dict of dataset featurespaces: {
                'id': ['Column_name_1', 'Column_name_2', ...]
                'binary': ['binary_column_name_1', 'binary_column_name_2', ...]
                'categorical': ['categorical_column_name_1', ...]
                'numerical': ['numerical_column_name_1', ...]
                'string': ['string_column_name_1', ...]
                'datetime': ['datetime_column_name_1', ...]
                }
        '''
        zero_time = time.time()

        self.f_spaces = {}
        nunique = {}

        for col in X.columns:
            nunique[col] = len(X[col].value_counts(dropna=False))
        nunique = pd.Series(nunique)
        self.f_spaces['constant'] = list(X.columns[nunique == 1])

        # Id_columns
        self.f_spaces['id'] = [col for col in X.columns if 'id_' in col]

        # Binary columns
        self.f_spaces['binary'] = list(X.columns[nunique == 2])
        self.f_spaces['binary'] = [col for col in self.f_spaces['binary'] if col != 'target']
        self.f_spaces['binary'] = [col for col in self.f_spaces['binary'] if\
                                   (col not in self.f_spaces['constant']) &\
                                   (col not in self.f_spaces['id'])]

        # String columns
        self.f_spaces['string'] = [col for col in X.columns if 'string_' in col]
        self.f_spaces['string'] = [col for col in self.f_spaces['string'] if\
                                   (col not in self.f_spaces['binary']) &\
                                   (col not in self.f_spaces['constant'])]

        # Datetime columns
        self.f_spaces['datetime'] = [col for col in X.columns if 'datetime_' in col]
        self.f_spaces['datetime'] = [col for col in self.f_spaces['datetime'] if\
                                     (col not in self.f_spaces['constant']) &\
                                     (col not in self.f_spaces['binary'])]

        # Categorical columns
        self.f_spaces['categorical'] = list(X.columns[(nunique > 2) &\
                                       (nunique <= self.OHE_MAX_VALUES)])
        self.f_spaces['categorical'] = [col for col in self.f_spaces['categorical']\
                                        if col != 'target']
        self.f_spaces['categorical'] = [col for col in self.f_spaces['categorical'] if\
                                        (col not in self.f_spaces['constant']) &\
                                        (col not in self.f_spaces['id']) &\
                                        (col not in self.f_spaces['binary']) &\
                                        (col not in self.f_spaces['datetime']) &\
                                        (col not in self.f_spaces['string'])]

        # Numerical columns
        self.f_spaces['number'] = [col for col in X.columns if\
                                   ('number_' in col) &\
                                   (col not in self.f_spaces['constant']) &\
                                   (col not in self.f_spaces['binary']) &\
                                   (col not in self.f_spaces['categorical'])]
        if self.verbose:
            print('FEATURE SPACE DEFINER: completed in {:.2f} sec'.format(time.time() - zero_time))
        return self.f_spaces


class ConstantColumnsRemover():
    '''
    Class that identifies and removes constan columns.

    Instance of this class that:
    - Identifies constant columns in train dataset
    - Stores column names that should be kept and dropped
    - Apply this transformation to the test dataset

    Args:
        f_spaces: dict with feature spaces description
        verbose: boolean, default False
            If True some information during processing will be output
    '''
    def __init__(self,
                 f_spaces: dict,
                 verbose: bool = False):
        self.f_spaces = f_spaces
        self.nunique = {}
        self.columns_to_drop = []
        self.verbose = verbose


    def fit(self, X: pd.DataFrame):
        '''
        Method identifies constant columns in train dataset.

        Args:
            X: train dataset

        Returns:
            An instance of self
        '''
        zero_time = time.time()

        # Count unique values in columns
        for col in X.columns:
            self.nunique[col] = len(X[col].value_counts(dropna=False))
        self.nunique = pd.Series(self.nunique)
        self.columns_to_drop = list(X.columns[self.nunique == 1])

        # Update feature spaces
        for space, cols in self.f_spaces.items():
            self.f_spaces[space] = [col for col in cols if col not in self.columns_to_drop]

        if self.verbose:
            print('GC: {} constant columns dropped: {}'.format(len(self.columns_to_drop),
                                                               self.columns_to_drop))
            print('GC: completed in {:.2f} sec'.format(time.time() - zero_time))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method removes constant columns in given dataset.

        Args:
            X: dataset to remove constant columns from

        Returns:
            Dataframe with no constant columns
        '''
        if len(self.columns_to_drop) > 0:
            X.drop(self.columns_to_drop, axis=1, inplace=True)
        return X


    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method applies fit and predict methods to given dataframe

        Args:
            X: dataset to remove constant columns from

        Returns:
            Dataframe with no constant columns
        '''
        self.fit(X)
        return self.transform(X)


class BasicFeaturePreprocessor():
    '''
    Class that provides features preprocessing:
        - unifies all nan values
        - converts strings to lower case
        - encodes id, binary and categorical columns with their frequency rank
        - reduces memory footprint
        - replaces inf values with maximum value in a column
        - replaces -inf values with minimum value in a column
        - encodes target column with int8 for binary classification task

    Args:
        mode: trainning mode for the model. Could be either 'regression' or 'classification'.
            All other inputs are considered as 'classification' problem.
        f_spaces: dict with feature spaces description
        task_type: some machine learning task type properties to provide better learning strategy
        ohe_max_values: max number of unique values in a column to consider it as categorical
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self,
                 mode: str,
                 f_spaces: dict,
                 task_type: dict,
                 ohe_max_values: int,
                 verbose=False):
        self.mode = mode
        self.f_spaces = f_spaces
        self.task_type = task_type
        self.OHE_MAX_VALUES = ohe_max_values

        self.id_features = f_spaces['id'].copy()
        self.num_features = f_spaces['number'].copy()
        self.bin_features = f_spaces['binary'].copy()
        self.dt_features = f_spaces['datetime'].copy()
        self.string_features = f_spaces['string'].copy()
        self.cat_features = f_spaces['categorical'].copy()

        self.id_value_encoder = {}
        self.bin_value_encoder = {}
        self.string_value_encoder = {}

        self.columns_to_drop = []
        self.verbose = verbose


    def preprocess_string_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess string columns:
            - unifies nan values
            - convests string to lower case
            - encodes string with their frequency rank

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        for col in self.string_features:
            X[col] = X[col].map(str).map(str.lower)
            X[col] = X[col].fillna('nan')
            X[col] = X[col].replace({'unknown': 'nan', 'na': 'nan',
                                     'n/a': 'nan', 'none': 'nan', '': 'nan'})
            X[col] = X[col].map(self.string_value_encoder[col])
            X[col] = X[col].fillna(len(self.string_value_encoder[col]) + 1).astype(np.uint32)
        return X


    def preprocess_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess categorical columns:
            - does nothing

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        return X


    def preprocess_datetime_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess datetime columns:
            - converts datetime columns to pandas datetime format

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        for col in self.dt_features:
            # Convert to datetime format
            X[col] = pd.to_datetime(X[col], errors='coerce')
        return X

    def preprocess_binary_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess binary columns:
            - encodes with 0/1 values
            - encodes unseen values with -1

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        for col in self.bin_features:
            X[col] = X[col].map(self.bin_value_encoder[col]).fillna(-1).astype(np.int8)
        return X


    def preprocess_id_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess id columns:
            - encodes with their frequencies
            - encodes unseen values with -1

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        for col in self.id_features:
            X[col] = X[col].map(self.id_value_encoder[col]).fillna(-1).astype(np.int32)
        return X


    def preprocess_numerical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess numerical columns:
            - replaces inf values with maximum value in each column
            - replaces -inf values with minimum value in each column

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        for col in self.num_features:
            sorted_vals = X[col].sort_values()
            X[col] = X[col].replace({np.inf: sorted_vals.iloc[-2], -np.inf: sorted_vals.iloc[1]})
        return X


    def preprocess_target(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that preprocess target columns:
            - encodes target column with 0/1 for classification problem

        Args:
            X: dataset to precess

        Returns:
            Pandas dataframe with processed columns
        '''
        if self.mode != 'regression' and 'target' in X.columns:
            X['target'] = X['target'].astype(np.int8)
        return X


    def fit(self, X: pd.DataFrame):
        '''
        Learns all encodings for each column in a dataset

        Args:
            X: dataset to fit to

        Returns:
            An instance of self
        '''
        zero_time = time.time()

        self.id_value_encoder = {}
        self.bin_value_encoder = {}
        self.string_value_encoder = {}

        # Learn encodings for binary columns
        for col in self.bin_features:
            col_unique = list(X[col].value_counts(dropna=False).index)
            self.bin_value_encoder[col] = dict(zip(col_unique, range(len(col_unique))))

        # Learn encodings for id columns
        for col in self.id_features:
            col_unique = list(X[col].value_counts(dropna=False).index)
            self.id_value_encoder[col] = dict(zip(col_unique, range(len(col_unique))))

        # Learn encodings for string columns
        for col in self.string_features:
            col_unique = X[col].value_counts(dropna=False).index[:self.OHE_MAX_VALUES-1]
            col_unique = col_unique.map(str).map(str.lower)
            self.string_value_encoder[col] = dict(zip(col_unique, range(len(col_unique))))
            self.f_spaces['categorical'] += [col]

        if self.verbose:
            eval_time = time.time() - zero_time
            print('FEATURE PREPROCESSOR: fit completed in {:.2f} sec'.format(eval_time))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Performs value encodings and preprocessing routines

        Args:
            X: dataset to generates features

        Returns:
            Preprocessed dataset
        '''
        zero_time = time.time()

        # Process datetime columns
        X = self.preprocess_datetime_columns(X)
        # Process id columns
        X = self.preprocess_id_columns(X)
        # Process binary columns
        X = self.preprocess_binary_columns(X)
        # Preprocess string features
        X = self.preprocess_string_columns(X)
        # Preprocess categorical features
        X = self.preprocess_categorical_columns(X)
        # Preprocess categorical features
        X = self.preprocess_numerical_columns(X)
        # Preprocess target
        X = self.preprocess_target(X)

        if self.verbose:
            eval_time = time.time() - zero_time
            print('FEATURE PREPROCESSOR: transform completed in {:.2f} sec'.format(eval_time))
        return X


    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method applies fit and transform methods to given dataframe

        Args:
            X: dataset to apply transformation to

        Returns:
            Transformed dataframe
        '''
        self.fit(X)
        return self.transform(X)


class FeatureGenerator():
    '''
    Class that provides new features generation.

    If dataset contains less than 3 datetime features the following time series features
    will be created: year, season, month, day, hour, minute, weekday, week hour, is_holiday,
    is_weekend, days before weekend, year month.

    If number of datetime feature more than one and less than 15 (to avoid feature explosion),
    all date differences will be created for each pair of original datetime features.

    Args:
        mode: trainning mode for the model. Could be either 'regression' or 'classification'.
            All other inputs are considered as 'classification' problem.
        f_spaces: dict with feature spaces description
        task_type: some machine learning task type properties to provide better learning strategy
        ohe_max_values: max number of unique values in a column to consider it as categorical
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self,
                 f_spaces: dict,
                 task_type: str,
                 mode: str = 'regression',
                 create_feature_interactions: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.f_spaces = f_spaces
        self.task_type = task_type
        self.create_nan_count = False
        self.new_datetime_features = []

        self.id_features = f_spaces['id'].copy()
        self.num_features = f_spaces['number'].copy()
        self.bin_features = f_spaces['binary'].copy()
        self.dt_features = f_spaces['datetime'].copy()
        self.string_features = f_spaces['string'].copy()
        self.cat_features = f_spaces['categorical'].copy()

        self.create_feature_interactions = create_feature_interactions
        self.div_interactions = []
        self.sum_interactions = []
        self.diff_interactions = []
        self.mult_interactions = []
        self.verbose = verbose


    def get_holiday_distance(self, date: np.datetime64) -> int:
        '''
        Method returns number of days before the nearest holidays

        Args:
            date: date to calculate distance

        Returns:
            Number of days before the nearest holidays
        '''
        distances = (self.holidays - date).astype('timedelta64[D]').astype(int)
        min_distance = min(distances[distances >= 0])
        return min(min_distance, 7)


    def generate_new_numerical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method adds to dataset new features generated based on numerical features.

        Generates following features:
            - number of nan counts in numerical features if it contains any information
            - all interactions of numerical features: sum, ratio, difference, multiplication

        Args:
            X: a dataset to add new features to

        Returns:
            Dataset with new generated features
        '''
        if self.create_nan_count:
            X['number_NANs_count'] = X.isnull().sum(axis=1)

        # Create feature interactions
        if self.create_feature_interactions:
            for col1, col2 in self.div_interactions:
                X['new_{}_to_{}'.format(col1, col2)] = X[col1]/(X[col2] + 1e-5)

            for col1, col2 in self.mult_interactions:
                X['new_{}_times_{}'.format(col1, col2)] = X[col1] * X[col2]

            for col1, col2 in self.diff_interactions:
                X['new_{}_diff_{}'.format(col1, col2)] = X[col1] - X[col2]

            for col1, col2 in self.sum_interactions:
                X['new_{}_plus_{}'.format(col1, col2)] = X[col1] + X[col2]
        return X


    def generate_new_binary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Stab method for consistency
        return X


    def generate_new_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Stab method for consistency
        return X


    def generate_new_id_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Stab method for consistency
        return X


    def generate_new_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method generates new datetime features generated based on datetime features.

        Args:
            X: a dataset to add new features to

        Returns:
            Dataset with new generated features
        '''
        days_before_next_weekend = {
            0: 5, 1: 4, 2: 3, 3: 2, 4: 1,
            5: 0, 6: 0
            }
        days_before_next_weekdays = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
            5: 2, 6: 1
            }
        month_to_season = dict(zip(range(1, 13), [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]))
        self.holidays = pd.to_datetime(pd.Series(ru_holidays))
        min_dt = X[self.dt_features].min().min()

        for col in self.dt_features:
            if self.task_type['is_timeseries'] or len(self.f_spaces['datetime']) < 3:
                X[col] = X[col].fillna(datetime.datetime(1970, 1, 1))

                X['TS_{}_year'.format(col)] = X[col].apply(lambda x: x.year).astype(np.int16)

                X['TS_{}_month'.format(col)] = X[col].apply(lambda x: x.month).astype(np.int8)

                X['TS_{}_day'.format(col)] = X[col].apply(lambda x: x.day).astype(np.int8)

                X['TS_{}_hour'.format(col)] = X[col].apply(lambda x: x.hour).astype(np.int8)

                X['TS_{}_minute'.format(col)] = X[col].apply(lambda x: x.minute).astype(np.int8)

                X['TS_{}_weekday'.format(col)] = \
                    X[col].apply(lambda x: x.weekday()).astype(np.int8)

                X['TS_{}_season'.format(col)] = \
                    X['TS_{}_month'.format(col)].map(month_to_season).astype(np.int8)

                X['TS_{}_hour_of_week'.format(col)] = \
                    X[col].apply(lambda x: x.weekday() * 24).astype(np.int16)

                X['TS_{}_is_holiday'.format(col)] = \
                    X[col].dt.date.isin(self.holidays.dt.date).astype(np.int8)

                X['TS_{}_is_weekend'.format(col)] = \
                    X['TS_{}_weekday'.format(col)].map({5: 1, 6: 1}).fillna(0).astype(np.int8)

                X['TS_{}_is_weekend'.format(col)] = (X['TS_{}_is_weekend'.format(col)] +\
                    X['TS_{}_is_holiday'.format(col)]).astype(np.int8)

                X['TS_{}_days_before_weekend'.format(col)] = \
                    X['TS_{}_weekday'.format(col)].map(days_before_next_weekend).astype(np.int8)

                year_month_func = lambda x: (x.year - 2010) * 12 + x.month
                X['TS_{}_year_month'.format(col)] = X[col].apply(year_month_func).astype(np.int16)

        if self.verbose:
            n_created = len([col for col in X.columns if 'TS_' in col])
            print('FEATURE GENERATOR: {} timeseries datetime features created'.format(n_created))

        for col1, col2 in self.new_datetime_features:
            X['new_datetime_diff' + col1 + '_' + col2] = \
                (X[col1] - X[col2]).astype('timedelta64[D]').fillna(-1).astype(np.int16)

        if self.verbose:
            n_created = len(self.new_datetime_features)
            print('FEATURE GENERATOR: {} datetime diff features created'.format(n_created))
        X.drop(self.f_spaces['datetime'], axis=1, inplace=True)
        return X


    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method that generates:
            - numerical feature interactions: sum, diff, mult and division (if required)
            - generates Nans_count features (number of Nan values in row) if it contains any
              information (Nans_count is nt constant feature).
            - time series features, like month, day, hour, etc.
            - generates dates differences which interaction gives more than 75% not Nan values

        Args:
            X: dataset to transform

        Returns:
            Transformed pandas dataframe
        '''
        zero_time = time.time()
        #
        if self.create_feature_interactions:
            self.div_interactions = []
            self.mult_interactions = []
            self.diff_interactions = []
            self.sum_interactions = []
            self.corr_threshold = 0.5
            correlations = [np.corrcoef(X['target'], X[col])[0, 1] for col in self.num_features]
            correlations = pd.Series(np.abs(correlations), index=self.num_features)
            candidates = np.array(self.num_features)[correlations < self.corr_threshold]
            corr_coef = 1.05

            for col1, col2 in permutations(self.num_features, 2):
                candidate = X[col1]/(X[col2] + 1e-5)
                correlation = np.abs(np.corrcoef(X['target'], candidate)[0, 1])
                if (correlation > max(correlations[col1], correlations[col2]) * corr_coef) and \
                    (correlation > self.corr_threshold):
                    self.f_spaces['number'] += ['new_{}_to_{}'.format(col1, col2)]
                    self.div_interactions += [(col1, col2)]
                    if self.verbose:
                        print('Create {} diff {} with corr factor={}'.format(col1, col2, correlation))

            for col1, col2 in combinations(self.num_features, 2):
                candidate = X[col1] * X[col2]
                correlation = np.abs(np.corrcoef(X['target'], candidate)[0, 1])
                if (correlation > max(correlations[col1], correlations[col2]) * corr_coef) and \
                   (correlation > self.corr_threshold):
                    self.f_spaces['number'] += ['new_{}_times_{}'.format(col1, col2)]
                    self.mult_interactions += [(col1, col2)]
                    if self.verbose:
                        print('Create {} times {} with corr factor={}'.format(col1, col2, correlation))

            for col1, col2 in combinations(self.num_features, 2):
                candidate = X[col1] - X[col2]
                correlation = np.abs(np.corrcoef(X['target'], candidate)[0, 1])
                if (correlation > max(correlations[col1], correlations[col2]) * corr_coef) and \
                   (correlation > self.corr_threshold):
                    self.f_spaces['number'] += ['new_{}_diff_{}'.format(col1, col2)]
                    self.diff_interactions += [(col1, col2)]
                    if self.verbose:
                        print('Create {} diff {} with corr factor={}'.format(col1, col2, correlation))

            for col1, col2 in combinations(self.num_features, 2):
                candidate = X[col1] + X[col2]
                correlation = np.abs(np.corrcoef(X['target'], candidate)[0, 1])
                if (correlation > max(correlations[col1], correlations[col2]) * corr_coef) and \
                   (correlation > self.corr_threshold):
                    self.f_spaces['number'] += ['new_{}_plus_{}'.format(col1, col2)]
                    self.sum_interactions += [(col1, col2)]
                    if self.verbose:
                        print('Create {} plus {} with corr factor={}'.format(col1, col2, correlation))

            if self.verbose:
                n_created = len(self.div_interactions + self.mult_interactions +
                                self.diff_interactions + self.sum_interactions)
                print('FEATURE GENERATOR: {} feature interactions created'.format(n_created))

        # Save new features names to f_spaces
        for col in self.dt_features:
            if self.task_type['is_timeseries'] or len(self.f_spaces['datetime']) < 3:
                self.f_spaces['categorical'].append('TS_{}_weekday'.format(col))
                self.f_spaces['categorical'].append('TS_{}_year'.format(col))
                self.f_spaces['categorical'].append('TS_{}_season'.format(col))
                self.f_spaces['categorical'].append('TS_{}_month'.format(col))
                self.f_spaces['categorical'].append('TS_{}_day'.format(col))
                self.f_spaces['categorical'].append('TS_{}_hour'.format(col))
                self.f_spaces['number'].append('TS_{}_year_month'.format(col))
                self.f_spaces['categorical'].append('TS_{}_hour_of_week'.format(col))
                self.f_spaces['categorical'].append('TS_{}_minute'.format(col))
                self.f_spaces['binary'].append('TS_{}_is_weekend'.format(col))
                self.f_spaces['binary'].append('TS_{}_is_holiday'.format(col))
                self.f_spaces['number'].append('TS_{}_days_before_weekend'.format(col))

        # Evaluate a list of datediff features
        # Take only feature differences which give more than 75% not Nans values
        self.new_datetime_features = []
        if len(self.dt_features) > 1 and len(self.dt_features) < 15:
            for col1, col2 in list(combinations(self.dt_features, 2)):
                if ((X[col1].notnull()) & (X[col2].notnull())).mean() > 0.5:
                    self.new_datetime_features.append((col1, col2))
                    self.f_spaces['number'].append('new_datetime_diff' + col1 + '_' + col2)

        # Check if sum of nans will contain any information
        if (X.isnull().sum(axis=1)).nunique() != 1:
            self.create_nan_count = True
            self.f_spaces['number'] += ['number_NANs_count']

        if self.verbose:
            eval_time = time.time() - zero_time
            print('FEATURE GENERATOR: fit completed in {:.2f} sec'.format(eval_time))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Generates new features for a given dataset.

        Args:
            X: dataset to transform

        Returns:
            Dataset with newly generated features
        '''
        zero_time = time.time()

        # Generate new features in binary feature space
        X = self.generate_new_binary_features(X)
        # Generate new features in numerical feature space
        X = self.generate_new_numerical_features(X)
        # Generate new features in categoriacl feature space
        X = self.generate_new_categorical_features(X)
        # Generate new features in datetime feature space
        X = self.generate_new_datetime_features(X)
        # Generate new features in id feature space
        X = self.generate_new_id_features(X)

        if self.verbose:
            eval_time = time.time() - zero_time
            print('FEATURE GENERATOR: transform completed in {:.2f} sec'.format(eval_time))
            print('FEATURE GENERATOR: X shape after feature generator {}'.format(X.shape))
        return X


    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method applies fit and transform methods to given dataframe

        Args:
            X: dataset to apply transformation to

        Returns:
            Transformed dataframe
        '''
        self.fit(X)
        return self.transform(X)


class TimeSeriesReverseOHE():
    '''
    Looking for one hot encoded time series features in a dataset and removes them.

    It is supposed that you have time series problem with main dt feature datetime_0 and
    some other features obtained by one hot encoding derivatirve features from datetime_0
    like day, month, week, etc.

    Args:
        f_spaces: dict with feature spaces description
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    # Apply for X without correlated features
    def __init__(self,
                 f_spaces: dict,
                 verbose: bool = False):
        self.f_spaces = f_spaces
        self.verbose = verbose
        self.columns_to_drop = []


    def fit(self, X: pd.DataFrame):
        '''
        Learns which features are one hot encoded datetime features and are going to be removed.

        Args:
            X: train dataset to fit to

        Returns:
            An instance of self
        '''
        zero_time = time.time()

        in_columns = X.shape[1]
        dt = X[self.f_spaces['datetime'][0]]
        periods = {'weekday': dt.dt.weekday,
                   'day': dt.dt.day,
                   'week': dt.dt.week,
                   'month': dt.dt.month,
                   'quarter': dt.dt.quarter,
                   'hour': dt.dt.hour}

        self.columns_to_drop = []

        for period, ts in periods.items():
            ohe_frame = pd.get_dummies(ts, prefix='ts_dummy_{}'.format(period), prefix_sep='=')
            ohe_columns = ohe_frame.columns
            corr_matrix = pd.concat((X, ohe_frame), axis=1).corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            corr_columns = [column for column in upper.columns if any(upper[column] == 1)]
            corr_rows = [row for row in upper.index if any(upper.loc[row] == 1)]
            if len(corr_columns) > 0:
                self.columns_to_drop += corr_rows
        if self.verbose:
            eval_time = time.time() - zero_time
            print('REVERSE OHE: fit completed in {:.2f} sec'.format(eval_time))
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Removes features which are one hot encoded datetime features.

        Args:
            X: train dataset to remove one hot encoded datetime features

        Returns:
            Dataset with removed one hot encoded datetime features
        '''
        zero_time = time.time()

        # Update feature spaces - remove correlated features
        for space, features in self.f_spaces.items():
            self.f_spaces[space] = [col for col in features if col not in self.columns_to_drop]

        if self.verbose:
            eval_time = time.time() - zero_time
            print('REVERSE OHE: {} ohed columns removed {}'.format(len(self.columns_to_drop),
                                                                   self.columns_to_drop))
            print('REVERSE OHE: transform completed in {:.2f} sec'.format(eval_time))
        return X[X.columns.difference(self.columns_to_drop)]


    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method applies fit and transform methods to given dataframe

        Args:
            X: dataset to apply transformation to

        Returns:
            Transformed dataframe
        '''
        self.fit(X)
        return self.transform(X)


class MeanTargetEncoderSmoothed():
    '''
    Performs smoothed mean target encoding for id, string and timeseries feature spaces.

    Args:
        f_spaces: dict with feature spaces description
        task_type: some machine learning task type properties to provide better learning strategy
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.

    '''
    def __init__(self,
                 f_spaces: dict,
                 task_type: dict,
                 verbose: bool = False):
        self.encoding_values = {}
        self.f_spaces = f_spaces
        self.features_to_encode = f_spaces['string'].copy()[:50]
        self.global_mean = None
        self.means_encoding = {}
        self.task_type = task_type
        self.verbose = verbose
        self.group_columns = {}


    def transform_train(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method learns encodings from train dataset and apply then apply it to
        id, string and timeseries feature spaces in dataset.

        Args:
            X: test dataset

        Returns:
            X: dataset with new mean target encoded features
        '''
        zero_time = time.time()

        self.features_to_encode += self.f_spaces['id'].copy()
        self.features_to_encode += [col for col in X.columns if 'TS_' in col]

        self.global_mean = np.mean(X['target'])
        # Smothing Hyperparameter
        min_samples_leaf = max(int(0.01 * len(X)), 20)
        smoothing = 0.5 * min_samples_leaf

        for col in self.features_to_encode:
            # Compute average for each column
            averages = X[[col, 'target']].groupby(col)['target'].agg(['mean', 'count'])

            # Compute smothed average
            self.smooth = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
            averages['target'] = self.global_mean * (1 - self.smooth) +\
                                 averages['mean'] * self.smooth
            # Store mapping 
            self.means_encoding[col] = averages['target'].to_dict()

            # Encode train dataset column
            encoded_col = X[col].map(self.means_encoding[col])
            X['mean_encode_{}'.format(col)] = encoded_col

            # Add new feature to feature space
            self.f_spaces['number'] += ['mean_encode_{}'.format(col)]

        if self.verbose:
            eval_time = time.time() - zero_time
            print('MEAN ENCODER: {} columns encoded'.format(len(self.features_to_encode)))
            print('MEAN ENCODER: train transform completed in {:.2f} sec'.format(eval_time))
        return X


    def transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method adds mean target encoded features to test dataset.

        Args:
            X: test dataset

        Returns:
            X: dataset with new mean target encoded features
        '''
        zero_time = time.time()
        for col in self.features_to_encode:
            X['mean_encode_{}'.format(col)] = X[col].map(self.means_encoding[col])
            X['mean_encode_{}'.format(col)].fillna(self.global_mean, inplace=True)

        if self.verbose:
            eval_time = time.time() - zero_time
            print('MEAN ENCODER: {} columns encoded'.format(len(self.features_to_encode)))
            print('MEAN ENCODER: test transform completed in {:.2f} sec'.format(eval_time))
        return X
