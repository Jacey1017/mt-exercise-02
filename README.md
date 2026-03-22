# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/Jacey1017/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download the dataset [AG News subset](https://huggingface.co/datasets/r-three/ag_news_subse) from HuggingFace:

    from datasets import load_dataset
    dataset = load_dataset("r-three/ag_news_subset")

Preprocess the dataset:

    python scripts/preprocess.py

This script also includes the download process, generating data/train.txt, data/valid.txt, data/test.txt.

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved to models/model.pt.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

# Dropout Experiments

This repo extends the original code to evaluate the effect of dropout regularization on language model perplexity. The updates are as follows:

A new command-line argument `-- log_file` was added to `tools/pytorch-examples/word_language_model/main.py`, as well as changes in the `train()` function. The script saves log files for the training and validation perplexities per epoch and the end of training test perplexity in a tabular format.

To run the experiments and train five models (dropout=0, 0.2, 0.4, 0.6, 0.8):
    
    ./scripts/train_dropouts.sh

Each model is saved to models/model_<dropout>.pt and its perplexity log is saved to logs/log_<dropout>.txt.

To visualize the results from previous log files:

    python scripts/results.py
Before running, make sure the `base_path` points to your local root.

This script prints a table of validation perplexity per epoch and test perplexity for each dropout setting and saves it as results/valid_perplexity_table.csv. Meanwhile, it generates a line chart per dropout to results/ppl_dropout_<value>.png, showing the train and validation perplexity over each epoch with a reference line of the final test perplexity. 