import os
import time
import pickle
import argparse

import numpy as np
import pandas as pd

from automl_utils.automodels import AutoModel


np.warnings.filterwarnings('ignore')
os.environ['JOBLIB_TEMP_FOLDER'] = '.'

# Use to read only first 2000 rows of each dateset
DEBUG_MODE = False

# Use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    if DEBUG_MODE:
        X = pd.read_csv(args.train_csv, nrows=50000)
    else:
        file_name = args.train_csv.name
        # Read just first 50 rows and look for datetime columns names to parse
        # them accordingly and reduce memory consuption
        X = pd.read_csv(file_name, encoding='utf-8', nrows=5000, low_memory=False)
        dtypes = X.dtypes.map(lambda x: 'float32' if x == 'float64' else x).to_dict()
        datetime_cols = X.columns[X.columns.str.contains('datetime_')].tolist()

        X = pd.read_csv(file_name,
                        encoding='utf-8',
                        dtype=dtypes,
                        parse_dates=datetime_cols,
                        infer_datetime_format=True,
                        skipinitialspace=True,
                        low_memory=False)

    print('==== Dataset read, shape {} ===='.format(X.shape))

    model = AutoModel(init_time=start_time, mode=args.mode, verbose=True)
    model.fit(X)

    model_filename = os.path.join(args.model_dir, 'model.pkl')
    with open(model_filename, 'wb') as fout:
        pickle.dump(model, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Total training time: {:.2f}'.format(time.time() - start_time))
