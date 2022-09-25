#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

mkdir -vp output/cnn_pipe

# set up the team1_f20_deep folder
cd team1_f20_deep
bash setup_cnn.sh
cd ..

# set up the team2_f20_wa folder
cd team2_f20_wa
bash setup_wa.sh
cd ..

