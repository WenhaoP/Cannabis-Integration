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


If you do not have a GPU available on your laptop or PC, remove the installation of `cudatoolkit` and `cudnn` packages in `team1_f20_deep/environment.yml`.

# Reproduction

To run the integration repository, run `run_all.sh` in the terminal.

# Shell scripts

There are many shell scripts for different automation purposes.

- `run_all.sh`:
- `clean.sh`:
- `setup.sh`:
- `team1_f20_deep/`
    - `clean_deep.sh`: 
    - `setup_deep.sh`: