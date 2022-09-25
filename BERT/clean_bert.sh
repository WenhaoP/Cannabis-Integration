#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base
mamba env remove -n deep_wa_bert
conda deactivate

rm data/full_dataset_with_labels.csv data/clean*.csv data/train.csv data/val.csv