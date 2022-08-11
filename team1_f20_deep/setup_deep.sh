#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base
mamba env create -f environment.yml
conda deactivate

# set up the team1_f20_deep folder
mkdir -vp metrics/validation/with_stop_words metrics/validation/without_stop_words
mkdir -vp metrics/test/with_stop_words metrics/test/without_stop_words
