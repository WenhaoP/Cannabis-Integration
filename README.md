# Cannabis-Integration
Integration of three cannabis project repositories

# Reproduction setup

1. Make sure you first install `mamba` in the base environment by running
```{bash}
conda install mamba -n base -c conda-forge
```

2. For all the shell scripts that have the following lines at the beginning,
```{bash}
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh
```
assign `CONDA_BASE` to the return of `conda info | grep -i 'base environment'`. Otherwise, you may see the error message like `conda command not found`. See more details in the discussion [here](https://github.com/conda/conda/issues/7980#issuecomment-441358406).

3. Depending on your operation system, you might need to replace the command word `bash` with `.` in each shell script.

4. If you do not have a GPU available on your laptop or PC, remove the installation of `cudatoolkit` and `cudnn` packages in `team1_f20_deep/environment.yml`.

5. Download `full_dataset.csv` from [here](https://drive.google.com/file/d/1lw2jXELtp0ADLUpBYDMkRlNflGn_stDr/view?usp=sharing) to `team1_f20_deep/data`.

# Reproduction

To run the integration repository, run `run_all.sh` in the terminal.

# Shell scripts

There are many shell scripts for different automation purposes.

- `run_bert_pipe.sh`: use the BERT to generate outputs for Washington data pipeline
- `run_cnn_pipe.sh`: use the TextCNN to generate outputs for Washington data pipeline
- `clean_bert_pipe.sh`: clean up `BERT` and `team2_f20_wa/` repositories
- `clean_cnn_pipe.sh`: clean up `team1_f20_deep` and `team2_f20_wa/` repositories
- `setup_bert_pipe.sh`: set up `BERT` and `team2_f20_wa/` repositories
- `setup_cnn_pipe.sh`: set up `team1_f20_deep` and `team2_f20_wa/` repositories
- `BERT/`:
    - `clean_bert.sh`: clean up the intermediate and final outputs of the `BERT` repository
    - `setup_bert.sh`: set up the environment and folders of the `BERT` repository 
- `team1_f20_deep/`
    - `clean_cnn.sh`: clean up the intermediate and final outputs of the `team1_f20_deep` repository
    - `setup_cnn.sh`: set up the environment and folders of the `team1_f20_deep` repository
- `team2_f20_wa/`
    - `clean_wa.sh`： clean up the intermediate and final outputs of the `team2_f20_wa/` repository
    - `setup_wa.sh`: set up the environment and folders of the `team2_f20_wa/` repository
    - `run_wa.sh`: run the `team2_f20_wa/` repository

# Note

1. All the current shell scripts for automation purpose were designed for the current hyperparameter values. Changign the hyperparameter valus 