#!/bin/bash

scripts=$(dirname "$0")
base=$(cd "$(dirname "$0")/.." && pwd)

models="$base/models"
data="$base/data"
tools="$base/tools"
logs="$base/logs"

mkdir -p "$models"
mkdir -p "$models"

num_threads=4
device=""

# create varying dropout settings
dropouts=(0 0.2 0.4 0.6 0.8)

# iterate model training
for d in "${dropouts[@]}"
do
    echo "=============================="
    echo "Training with dropout=$d"
    echo "=============================="
    (cd "$tools/pytorch-examples/word_language_model" &&
        OMP_NUM_THREADS=$num_threads python main.py \
            --data "$data" \
            --epochs 40 \
            --log-interval 100 \
            --emsize 250 --nhid 250 --dropout $d --tied \
            --save "$models/model_$d.pt" \
            --log_file "$logs/log_$d.txt"
)
done
