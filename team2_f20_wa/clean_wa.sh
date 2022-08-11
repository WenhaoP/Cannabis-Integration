#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base
mamba env remove -n wa_pipeline_env
conda deactivate

# clean the team2_f20_wa folder
rm -rf Processed_Data/* Plots/* Raw_Data/full_dataset_with_labels.csv