#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

mkdir -vp output/bert_pipe

# set up the BERT folder
cd BERT
bash setup_bert.sh
cd ..

# set up the team2_f20_wa folder
cd team2_f20_wa
bash setup_wa.sh
cd ..