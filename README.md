# AutoML

## About
AutoML solution aimed at fitting to arbitrary datasets including automatic data cleaning/preprocessing, feature generation, hyper-parameters tuning, model selection etc. 
This solution was implemented during [Sberbank Data Science Journey 2018](https://sdsj.sberbank.ai/en/) contest.

## Data Structure
AutoML solution has to perform well on arbitrary datasets in CSV format.
There are only very basic requirements to data sctructure in csv file:
- `line_id` — an Id for each line
- `target` — target variable (only for train dataset), continuous variable for regression tasks and binary labels (0/1) for classification
- `<type>_<feature number>` — type of other features in a dataset, where type could be:
    - `id` — Id (special purpose categorical variable)
    - `datetime` — date feature in 2010-01-01 or 2010-01-01 10:10:10 format
    - `number` — number feature (also could be continuous, categorical or binary variable)
    - `string` — string feature

For instance:

|line_id |   id_0   | datetime_0 | number_0 | number_1 | number_2 | string_0 |                 string_1                 | 
| ------ | -------- | ---------- | -------- | -------- | -------- | -------- | ---------------------------------------- | 
|    0   | GUID-1209| 2018-01-01 |  1000.00 |    12    |  0.77777 |   male   | 1 New Orchard Rd, Armonk, New York 10504 | 


## Requirements
AutoML has the following requirements:
- Available resources:
  - **12 Gb** RAM
  - **4** vCPU
  - **1 Gb** of local disk space available for writing
- AutoML has no access to the Internet
- Dataset CSV file size is less than **3 Gb**
- AutoML solution should accept training time limit as a parameter. Default value is **5 min**.


## Evaluation
Optimization metric for regression task - RMSE.\
Optimization metric for binary classification task - ROC_AUC.


## Examples
### Set Training Time Limit
To set time limit for training process just set time limit to environmet variable **TIME_LIMIT** in second.\
E.g. to set training time limit to 300 sec:
```
export TIME_LIMIT=300
```
If TIME_LIMIT variable is not defined default value of **300 sec** will be used.

### Train model
To train regression model:\
```
python3 train.py \
    --mode regression \
    --train-csv <path_to_train_data_file> \
    --model-dir <model_directory_path>
```

To train classification task:\
```
python3 train.py \
    --mode classification \
    --train-csv <path_to_train_data_file> \
    --model-dir <model_directory_path>
```

`<path_to_train_data_file>` - path to a csv file you'd like to train your model on\
`<model_directory_path>` - path to a directory you want to save your model after it is trained

Example:
```
python3 train.py \
    --mode classification \
    --train-csv train.csv \
    --model-dir .
```


### Use model for prediction
To make a prediction for regression task:\
```
python3 predict.py \
    --test-csv <path_to_test_data_file> \
    --prediction-csv <path_to_prediction_file> \
    --model-dir <model_directory_path>
```

`<path_to_test_data_file>` - path to a csv file you'd like to make prediction for. Should have the same structure as train file does but without target vector\
`<path_to_prediction_file>` - path to a csv file you want to save the prediction to\
`<model_directory_path>` - path to a directory where your trained model is located

Example:\
```
python3 predict.py \
    --test-csv test.csv \
    --prediction-csv prediction.csv \
    --model-dir .
```


## Try on SDSJ 2018 Data
You can try this solution on the data provided during Sberbank Datascience Journey 2018.\
It contains 8 sets of data of different sizes (3 regression and 5 classification problems). Each set contains:
- train.csv - train dataset
- test.csv - test dataset
- test-target.csv - ground truth target vector for test data

To validate the model on all sets of data:
- Download [data](https://s3.eu-central-1.amazonaws.com/sdsj2018-automl/public/sdsj2018_automl_check_datasets.zip) (**348 Mb**).
- Extract data to validation_data folder in this repo. It requires **1.95 Gb** of dick space after extraction.
- Run run_validation_tests.sh script to train and validate on each of datasets.

After script is executed you will get a similar output:
```
1. Validation MSE:          72.24    Baseline MSE:         397.07    Relative model quality:    81.81%
2. Validation MSE:           1.62    Baseline MSE:           5.29    Relative model quality:    69.37%
3. Validation MSE: 11535683175.54    Baseline MSE: 41054362265.11    Relative model quality:    71.90%
4. Validation AUC:           1.00    Baseline AUC:           0.50    Relative model quality:   100.00%
5. Validation AUC:           0.78    Baseline AUC:           0.50    Relative model quality:    55.61%
6. Validation AUC:           0.66    Baseline AUC:           0.50    Relative model quality:    31.20%
7. Validation AUC:           0.74    Baseline AUC:           0.50    Relative model quality:    48.65%
8. Validation AUC:           0.89    Baseline AUC:           0.50    Relative model quality:    77.56%
=====================================================
Mean model quality at regression task: 74.36%
Mean model quality at classifiaction task: 62.60%
Mean model quality: 67.01%
```
Where:
- `Validation MSE/AUC` - model's validation score on test data (there is test target vector for each set of data)
- `Baseline MSE` - validation score of a constant prediction (mean target)
- `Baseline AUC` - validation score 0.5 of random generator prediction
- `Relative model quality` = (`Validation score` - `Baseline score`) / (`Best score` - `Baseline score`). 
- `Best score` is **0** for regression task and **1.0** for classification task.
