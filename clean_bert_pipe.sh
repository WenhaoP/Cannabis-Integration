#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

rm -rf output/bert_pipe/*

# clean the BERT folder
cd BERT
bash clean_bert.sh
cd ..

# clean the team2_f20_wa folder
cd team2_f20_wa
bash clean_wa.sh
cd ..
