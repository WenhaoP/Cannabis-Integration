#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base
mamba env remove -n deep_wa
mamba env remove -n wa_pipeline_env
conda deactivate

rm -rf output/*

# clean the team1_f20_deep folder
cd team1_f20_deep
rm -rf metrics/* images/f1_scores.png data/clean_in_sample.csv data/clean_out_sample.csv
cd ..

# clean the team2_f20_wa folder
cd team2_f20_wa
rm -rf Processed_Data/* Plots/* Raw_Data/full_dataset_with_labels.csv
cd ..