#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base
mamba env remove -n deep_wa
conda deactivate

# clean the team1_f20_deep folder
rm -rf metrics/* images/f1_scores.png data/clean_in_sample.csv data/clean_out_sample.csv
