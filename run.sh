#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base

bash clean.sh
bash setup.sh

conda deactivate

# run team1_f20_deep
cd team1_f20_deep
conda activate deep_wa
python main.py
conda deactivate
cd ..
cp team1_f20_deep/data/full_dataset_with_labels.csv team2_f20_wa/Raw_Data

# run team2_f20_wa
cd team2_f20_wa
conda activate wa_pipeline_env
bash run_wa.sh
conda deactivate
cd ..
cp team2_f20_wa/Processed_Data/pipeline_final_output.csv .