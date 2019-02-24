#!/bin/sh
DATA_DIR="./validation_data/"
MODEL_PATH="./"

# Test regression tasks
# export TIME_LIMIT=300
counter=1
while [ $counter -le 3 ]
do
	echo "=========================================="
	echo "============== Test case" $counter "==============="
	echo "=========================================="
	python3 train.py --mode regression --train-csv $DATA_DIR"check_"$counter"_r/train.csv" --model-dir $MODEL_PATH
	python3 predict.py --test-csv $DATA_DIR"check_"$counter"_r/test.csv" --prediction-csv $DATA_DIR"check_"$counter"_r/prediction.csv" --model-dir $MODEL_PATH
	((counter++))
done

# Test classification tasks
while [ $counter -le 8 ]
do
	echo "=========================================="
	echo "============== Test case" $counter "==============="
	echo "=========================================="
	python3 train.py --mode classification --train-csv $DATA_DIR"check_"$counter"_c/train.csv" --model-dir $MODEL_PATH
	python3 predict.py --test-csv $DATA_DIR"check_"$counter"_c/test.csv" --prediction-csv $DATA_DIR"check_"$counter"_c/prediction.csv" --model-dir $MODEL_PATH
	((counter++))
done

# Validate predictions
python3 score_validation.py

