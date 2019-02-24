import os
import time
import pickle
import argparse

import pandas as pd
import numpy as np

np.warnings.filterwarnings('ignore')

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv', type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    # Load model
    model_filename = os.path.join(args.model_dir, 'model.pkl')
    with open(model_filename, 'rb') as fin:
        model = pickle.load(fin)

    # Read dataset and make predictions by chunks
    chunk_size = model.df_read_chunk_size_
    file_name = args.test_csv.name
    X = pd.read_csv(file_name, encoding='utf-8', nrows=5000, low_memory=False)
    dtypes = X.dtypes.map(lambda x: 'float32' if x == 'float64' else x).to_dict()
    datetime_cols = X.columns[X.columns.str.contains('datetime')].tolist()

    result = {
        'line_id': [],
        'prediction': []
    }

    for X in pd.read_csv(file_name,
                         encoding='utf-8',
                         dtype=dtypes,
                         low_memory=False,
                         parse_dates=datetime_cols,
                         infer_datetime_format=True,
                         skipinitialspace=True,
                         chunksize=chunk_size):
        print('==== Dataset chunk read, shape {} ====='.format(X.shape))
        result['line_id'] += list(X['line_id'])
        print('Predicting on chunk, shape {}'.format(X.shape))
        result['prediction'] += list(model.predict(X)['prediction'])

    prediction_file_name = args.prediction_csv.name
    prediction_dframe = pd.DataFrame(result)

    prediction_dframe.to_csv(prediction_file_name, index=False)

    print('Total prediction time: {}'.format(time.time() - start_time))
