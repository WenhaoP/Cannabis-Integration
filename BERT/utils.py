####################################################
## Author: Wenhao Pan
## Institution: Berkeley Institute for Data Science
## Spring 2022, Summer 2022
####################################################

"""
Contains all functions used in text preprocessing
"""

import pandas as pd
import numpy as np
import os
from gensim.parsing import remove_stopwords, strip_numeric, strip_punctuation, strip_multiple_whitespaces

from params import *

def clean_data(df, field, labels, punctuations=True, stop_words=True, digits=True, minimal=False):
    """Binarizes labels for given dataframe, and exports cleaned dataframes

    Args:
        df (pd.dataframe): dataframe with label columns (see LABELS above)
        field (str): the name of the input field
        labels (list[str]): labels we currently consider
        punctuations (boolean): keep punctuations from the description field if True
        stop_words (boolean): keep stop words from the description field if True
        digits (boolean): keep digits from the description field if True
        minimal (boolean): only keep the description and label fields if True

    Returns:
        df_clean (pd.dataframe): cleaned dataframe with binarized labels
    """
    df_clean = df.dropna(subset=[field])

    # ensure label fields are all numerical
    if (not PREDICT):
        for label in FULL_LABELS:
            df_clean = df_clean[(df_clean[label] == 0) | (df_clean[label] == 1) | (df_clean[label] == '0') | (df_clean[label] == '1')]
            df_clean[label] = pd.to_numeric(df_clean[label])
    
    # remove punctuations if wanted
    if not punctuations:
        df_clean[field] = df_clean[field].apply(strip_punctuation)

    # remove stopwords if wanted 
    if not stop_words:
        df_clean[field] = df_clean[field].apply(remove_stopwords)
    
    # remove digits if wanted
    if not digits:
        df_clean[field] = df_clean[field].apply(strip_numeric)

    # drop unnecessary columns
    if minimal:
        df_clean = df_clean[[field] + list(set(FULL_LABELS) & set(labels))]

    df_clean[field] = df_clean[field].astype(str)
    df_clean[field] = df_clean[field].str.lower() # lowercase all characters
    df_clean[field] = df_clean[field].apply(strip_multiple_whitespaces) # remove repeating whitespace
    df_clean = df_clean.replace(to_replace=[''], value=np.nan).dropna(subset=[field]) # drop empty field
    
    return df_clean


def load_data(field, labels, punctuations=True, stop_words=True, digits=True, minimal=False):
    """Loads in_sample and out_sample data, cleans them, and exports clean csv files

    Args:
        field (str): the name of the input field
        labels (list[str]): labels we currently consider
        punctuations (boolean): keep punctuations from the description field if True
        stop_words (boolean): keep stop words from the description field if True
        digits (boolean): keep digits from the description field if True
        minimal (boolean): only keep the description and label fields if True

    Returns:
        clean_insample (pd.DataFrame): Training Dataset
        clean_outsample (pd.DataFrame): Testing Dataset
    """
    # Check that data is downloaded
    assert os.path.exists("data/in_sample.csv"), "Need to download in_sample.csv first!"
    assert os.path.exists("data/out_sample.csv"), "Need to download out_sample.csv first!"

    insample = pd.read_csv("data/in_sample.csv")
    clean_insample = clean_data(insample, field, labels, punctuations, stop_words, digits, minimal)

    outsample = pd.read_csv("data/out_sample.csv")
    clean_outsample = clean_data(outsample, field, labels, punctuations, stop_words, digits, minimal)

    # if ('Medical_Wellness' in labels):
    #     clean_insample['Medical_Wellness'] = np.logical_or(clean_insample['Medical'], clean_insample['Wellness']).astype(int)
    #     clean_outsample['Medical_Wellness'] = np.logical_or(clean_outsample['Medical'], clean_outsample['Wellness']).astype(int)
    
    # if ('Pre_Hybrid' in labels):
    #     Medical_Wellness = np.logical_or(clean_insample['Medical'], clean_insample['Wellness']).astype(int)
    #     clean_insample['Pre_Hybrid'] = np.logical_and(Medical_Wellness, clean_insample['Intoxication']).astype(int)

    #     Medical_Wellness = np.logical_or(clean_outsample['Medical'], clean_outsample['Wellness']).astype(int)
    #     clean_outsample['Pre_Hybrid'] = np.logical_and(Medical_Wellness, clean_outsample['Intoxication']).astype(int)

    clean_insample.to_csv('data/clean_in_sample.csv', index=False)
    clean_outsample.to_csv('data/clean_out_sample.csv', index=False) 

    return clean_insample, clean_outsample

# Define the preprocess function ###
def preprocess_function(examples):
    """
    Preprocess the description field
    ---
    Arguments:
    examples (str, List[str], List[List[str]]: the sequence or batch of sequences to be encoded/tokenized

    Returns:
    tokenized (transformers.BatchEncoding): tokenized descriptions 
    """
    tokenized = TOKENIZER(
        examples["straindescription"],
        padding=PADDING,
        truncation=TRUNCATION,
        max_length=MAX_LEN,
    )

    return tokenized