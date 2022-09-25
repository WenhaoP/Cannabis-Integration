#!/bin/bash
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh

conda activate base

printf "================ Started cleaning the integration repository ================\n\n"
bash clean_cnn_pipe.sh
printf "================ Finished cleaning the integration repository ================\n\n"

printf "================ Started setting up the integration repository ================\n\n"
bash setup_cnn_pipe.sh
printf "================ Finished setting up the integration repository ================\n\n"

conda deactivate

printf "================ Started running the team1_f20_deep repository ================\n\n"
cd team1_f20_deep
conda activate deep_wa_cnn
python main.py
conda deactivate
cd ..
printf "================ Finished running the team1_f20_deep repository ================\n\n"

printf "================ Started copying the output of team1_f20_deep to team2_f20_wa ================\n\n"
cp team1_f20_deep/data/full_dataset_with_labels.csv team2_f20_wa/Raw_Data
printf "================ Finished copying the output of team1_f20_deep to team2_f20_wa ================\n\n"

printf "================ Started running the team2_f20_wa repository ================\n\n"
cd team2_f20_wa
conda activate wa_pipeline_env
bash run_wa.sh
conda deactivate
cd ..
printf "================ Finished running the team2_f20_wa repository ================\n\n"

printf "================ Extracting the final output ================\n\n"
cp team2_f20_wa/Processed_Data/pipeline_final_output.csv output/cnn_pipe/
cp team1_f20_deep/data/full_dataset_with_labels.csv output/cnn_pipe/
cp team1_f20_deep/metrics/best_val_models.csv output/cnn_pipe/
cp team1_f20_deep/metrics/best_test_models.csv output/cnn_pipe/