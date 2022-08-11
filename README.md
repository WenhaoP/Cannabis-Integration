# Cannabis-Integration
Integration of three cannabis project repositories

# Reproduction setup

Make sure you first install `mamba` in the base environment by running
```{bash}
conda install mamba -n base -c conda-forge
```

For all the shell scripts that have the following lines at the beginning,
```{bash}
CONDA_BASE='C:\Users\Wenhao\miniconda3'
source $CONDA_BASE/etc/profile.d/conda.sh
```
assign `CONDA_BASE` to the return of `conda info | grep -i 'base environment'`. Otherwise, you may see the error message like `conda command not found`. See more details in the discussion [here](https://github.com/conda/conda/issues/7980#issuecomment-441358406).

Depending on your operation system, you might need to replace the command word `bash` with `.` in each shell script.

If you do not have a GPU available on your laptop or PC, remove the installation of `cudatoolkit` and `cudnn` packages in `team1_f20_deep/environment.yml`.

# Reproduction

To run the integration repository, run `run_all.sh` in the terminal.

# Shell scripts

There are many shell scripts for different automation purposes.

- `run_all.sh`: run all the repositories in order
- `clean_all.sh`: clean up the intermediate and final outputs of each repository
- `setup_all.sh`: set up the environment and folders of each repository
- `team1_f20_deep/`
    - `clean_deep.sh`: clean up the intermediate and final outputs of the `team1_f20_deep` repository
    - `setup_deep.sh`: set up the environment and folders of the `team1_f20_deep` repository
- `team2_f20_wa/`
    - `clean_wa.sh`： clean up the intermediate and final outputs of the `team2_f20_wa/` repository
    - `setup_wa.sh`: set up the environment and folders of the `team2_f20_wa/` repository
    - `run_wa.sh`: run the `team2_f20_wa/` repository