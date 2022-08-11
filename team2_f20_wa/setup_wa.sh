#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

# set up the team2_f20_wa folder
mkdir -vp Processed_Data Plots
conda activate base
mamba env create -f wa_pipeline_env.yml
conda deactivate