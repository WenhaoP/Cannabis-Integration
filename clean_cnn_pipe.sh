#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

rm -rf output/cnn_pipe/*

# clean the team1_f20_deep folder
cd team1_f20_deep
bash clean_cnn.sh
cd ..

# clean the team2_f20_wa folder
cd team2_f20_wa
bash clean_wa.sh
cd ..

