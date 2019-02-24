import numpy as np
import pandas as pd


class MemUsageOptimizer():
    '''
    Optimizes dataset size setting least memory demanding datatypes for each column

    Args:
        verbose: the verbosity parameter. If False, model outputs only genaral info regarding
            training and prediction processes.
    '''
    def __init__(self, verbose=False):
        self.dtypes = {}
        self.verbose = verbose


    def fit(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Finds optimal datatype for each columns in a dataset to reduce memory consuption

        Args:
            X: train dataset to fit to

        Returns:
            An instance of self
        '''
        for col in X.columns.difference(['target', 'line_id']):
            col_type = X[col].dtype

            if (col_type != object) and ('datetime' not in col) and \
                ('id_' not in col) and ('string_' not in col):
                c_min = X[col].min()
                c_max = X[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.dtypes[col] = np.int8
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.dtypes[col] = np.int16
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.dtypes[col] = np.int32
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.dtypes[col] = np.int64
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.dtypes[col] = np.float16
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.dtypes[col] = np.float32
                    else:
                        self.dtypes[col] = np.float64
        return X


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Changes dataset's column types to reduce memory consumption

        Args:
            X: dataframe needs to be optimized

        Returns:
            Optimized dataframe
        '''
        if self.verbose:
            original_mem_size = X.memory_usage(deep=True).sum() / 1024**2
            print('MEM OPTIMIZER: Memory usage of dataframe: {:.2f} MB'.format(original_mem_size))

        X = X.astype(self.dtypes, copy=False)

        if self.verbose:
            new_mem_size = X.memory_usage(deep=True).sum() / 1024**2
            print('MEM OPTIMIZER: Memory usage after optimization: {:.2f} MB'.format(new_mem_size))
            print('MEM OPTIMIZER: Decreased by {:.1f}%'.format(\
                100 * (original_mem_size - new_mem_size) / original_mem_size))
        return X


    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        '''
        Method applies fit and transform methods to given dataframe

        Args:
            X: dataset to apply optimization to

        Returns:
            Optimized dataframe
        '''
        self.fit(X)
        return self.transform(X)
