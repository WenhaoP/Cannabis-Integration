####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Summer 2021, Fall 2021
####################################################

"""
Contains all functions used in text preprocessing
"""

from params import *
from gensim.parsing.preprocessing import remove_stopwords
import pandas as pd
import os


def clean_data(df, field='straindescription', stop_words=True):
    """Binarizes labels for given dataframe, and exports cleaned dataframes

    Args:
        df (pd.dataframe): dataframe with label columns (see LABELS above)
        field (str): the textual field to clean
        stop_words (boolean): keep stop words from the description field if True; remove otherwise

    Returns:
        df_clean (pd.dataframe): cleaned dataframe with binarized labels
    """
    df_clean = df.dropna(subset=[field])

    # ensure label fields are all numerical
    for label in LABELS:
        df_clean = df_clean[(df_clean[label] == 0) | (df_clean[label] == 1) | (df_clean[label] == '0') | (df_clean[label] == '1')]
        df_clean[label] = pd.to_numeric(df_clean[label])
    
    # remove stop words if wanted 
    if not stop_words:
        df_clean[field] = df_clean[field].apply(remove_stopwords)

    return df_clean


def load_data(stop_words=True):
    """Loads in_sample and out_sample data, cleans them, and exports clean csv files

    ** Data files are too large to store on github, so they must be downloaded to ~/data/ locally before running

    Args:
        stop_words (boolean): keep stop words from the description field if True; remove otherwise

    Returns:
        clean_insample (pd.DataFrame): Training Dataset
        clean_outsample (pd.DataFrame): Testing Dataset
    """
    # Check that data is downloaded
    assert os.path.exists("data/in_sample.csv"), "Need to download in_sample.csv first!"
    assert os.path.exists("data/out_sample.csv"), "Need to download out_sample.csv first!"

    if stop_words:
        folder = 'with_stop_words'
    else:
        folder = 'without_stop_words'

    insample = pd.read_csv("data/in_sample.csv")
    clean_insample = clean_data(insample, stop_words=stop_words)
    clean_insample.to_csv(f'data/clean_in_sample_{folder}.csv', index=False)

    outsample = pd.read_csv("data/out_sample.csv")
    clean_outsample = clean_data(outsample, stop_words=stop_words)
    clean_outsample.to_csv(f'data/clean_out_sample_{folder}.csv', index=False)

    return clean_insample, clean_outsample


def find_csv_filenames(path_to_dir, suffix=".csv"):
    """Find all files within a specific directory that are of a certain filetype

    Args:
        path_to_dir (str, os.path): directory to search through for files
        suffix (str, optional): type of file to search for. Defaults to ".csv".

    Returns:
        (list)): list of all files in the directory that match the filetype
    """
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]