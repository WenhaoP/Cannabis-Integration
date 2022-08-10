#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

# set up the team1_f20_deep folder
cd team1_f20_deep
mkdir -vp metrics/validation/with_stop_words metrics/validation/without_stop_words
mkdir -vp metrics/test/with_stop_words metrics/test/without_stop_words
mamba env create -f environment.yml
cd ..

# set up the team2_f20_wa folder
cd team2_f20_wa
mkdir -vp Processed_Data Plots
mamba env create -f wa_pipeline_env.yml
cd ..