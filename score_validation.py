import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error


rmse_history = []
roc_auc_history = []
model_quality_history_regression = []
model_quality_history_classification = []

DATA_PATH = './validation_data/'

for i in range(1,4):
    train = pd.read_csv(DATA_PATH + 'check_{}_r/train.csv'.format(i), low_memory=False)
    mean_prediction = train['target'].mean()

    predicted = pd.read_csv(DATA_PATH + 'check_{}_r/prediction.csv'.format(i))
    y_test = pd.read_csv(DATA_PATH + 'check_{}_r/test-target.csv'.format(i))
    predicted.sort_values(['line_id'], inplace=True)
    y_test.sort_values(['line_id'], inplace=True)
    score = mean_squared_error(y_test['target'], predicted['prediction'])
    baseline_score = mean_squared_error(y_test['target'], np.ones_like(y_test['target']) * mean_prediction)
    rmse_history.append(score)
    model_quality_history_regression.append((1 - score/baseline_score)*100)
    print('{}. Validation MSE: {:16.2f}    Baseline MSE: {:16.2f}    Relative model quality: {:8.2f}%'.format(i, score, baseline_score, 
                                                                                            (1 - score/baseline_score)*100))
for i in range(4,9):
    predicted = pd.read_csv(DATA_PATH + 'check_{}_c/prediction.csv'.format(i))
    y_test = pd.read_csv(DATA_PATH + 'check_{}_c/test-target.csv'.format(i))
    score = roc_auc_score(y_test['target'], predicted['prediction'])
    roc_auc_history.append(score)
    model_quality_history_classification.append((score/0.5 - 1)*100)
    print('{}. Validation AUC: {:16.2f}    Baseline AUC: {:16.2f}    Relative model quality: {:8.2f}%'.format(i, score, 0.5, 
                                                                                            ((score/0.5 - 1)*100)))

print('=====================================================')
print('Mean model quality at regression task: {:.2f}%'.format(np.mean(model_quality_history_regression)))
print('Mean model quality at classifiaction task: {:.2f}%'.format(np.mean(model_quality_history_classification)))
print('Mean model quality: {:.2f}%'.format(np.mean(model_quality_history_regression + model_quality_history_classification)))