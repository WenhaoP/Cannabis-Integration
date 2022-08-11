####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Summer 2021, Fall 2021
####################################################

"""
Contains all functions used in model tuning and evaluation of trained models
"""
from utils import *
from params import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix


def get_metrics(X, y, model, num_filters, kernel_size, dilation_rate, vocab_size, embedding_dim, maxlen, validation_size):
    """Creates a dictionary containing all of the hyperparameters and the resulting evaluation metrics for a trained model

    Args:
        X (pd.dataframe, np.array): Design matrix that gets passed into trained model
        y (pd.series, np.array): Label vector
        model (keras model object): trained model to be evaluated
        num_filters (int): the number of filters per filter size
        kernel_size (int): the number of words we want our convolutional filters to cover. 
                           We will have num_filters for each size specified here. For example, 
                           [2, 3] means that we will have filters that slide over 2 and 3 words respectively.
        dilation_rate (int): skip size of the convolution on the embedding
        vocab_size (int): the size of our vocabulary. This is needed to define the size of our embedding layer, 
                          which will have shape [vocabulary_size, embedding_size]
        embedding_dim (int): the dimensionality of our embeddings
        maxlen (int): maximum number of words kept in the product description. Padding will be added if description
                      is shorter than maxlen
        validation_size (float): size of the validation set, (0, 1)

    Returns:
        metrics (dictionary): keys are model hyperparemeters and evaluation metrics, with associated corresponding values
    """
    predictions = model.predict(X)
    predictions = predictions.round()

    metrics = {}

    metrics['pred_true'], metrics['pred_false'] = np.count_nonzero(predictions), predictions.shape[0] - np.count_nonzero(predictions)
    metrics['actual_true'], metrics['actual_false'] = np.count_nonzero(y), y.shape[0] - np.count_nonzero(y)
    metrics['accuracy'] = accuracy_score(y, predictions)
    metrics['precision'] = precision_score(y, predictions)
    metrics['recall'] = recall_score(y, predictions)
    metrics['f1'] = f1_score(y, predictions)

    metrics['matthews_correlation'] = matthews_corrcoef(y, predictions)
    metrics['cohen_kappa'] = cohen_kappa_score(y, predictions)
    metrics['roc_auc'] = roc_auc_score(y, predictions)    

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    metrics['specificity'] = tn / (tn+fp)
    metrics['sensitivity'] = tp / (tp+fn)
    metrics['informedness'] = metrics['specificity'] + metrics['sensitivity'] - 1

    metrics['tn'] = tn
    metrics['fp'] = fp
    metrics['fn'] = fn
    metrics['tp'] = tp

    metrics['num_filters'] = num_filters
    metrics['kernel_size'] = kernel_size
    metrics['dilation'] = dilation_rate
    metrics['vocab_size'] = vocab_size
    metrics['embedding_dim'] = embedding_dim
    metrics['maxlen'] = maxlen
    metrics['validation_set_size'] = validation_size
        
    return metrics


def find_best_model(data, metric="f1", stop_words=True):
    """Iterates through all results on the validation data to find optimal results

    Args:
        data(string): either "validation" or "test"
        metric (string): column in the validation dataframe that is used to rank the model results

    Returns:
        best_models (pd.DataFrame): Table containing the top models according to metric
            - Can tune how many models are returned with NUM_BEST_MODELS in params.py
    
    """
    if stop_words:
        folder = 'with_stop_words'
    else:
        folder = 'without_stop_words'

    # Load Validation results
    files = []
    directory_path = "metrics/" + data + "/" + folder + "/"
    filenames = find_csv_filenames(directory_path)
    for name in filenames:
        filename = directory_path + name
        df = pd.read_csv(filename)
        files.append(df)
    df_data = pd.concat(files)

    # Rank results by output class and store in new dataframe
    results = []
    for label in LABELS + BALANCED_LABELS:
        results.append(df_data[df_data['model_name'] == label].sort_values(by=metric, ascending=False).iloc[:NUM_BEST_MODELS])
    best_models = pd.concat(results)
    best_models['rank'] = list(range(1, NUM_BEST_MODELS+1)) * (int(len(best_models) / NUM_BEST_MODELS))
    
    return best_models


def plot_results(data, metric, hue, ci="sd", palette="dark", alpha=0.6, height=10, ylim=(0.7, 1)):
    """Plots seaborn catplot of given metric across all label categories

    Args:
        data (pd.dataframe): Dataframe containing the results of a model
        metric (string): specific metric to visualize (see get_metrics() for reference)
        hue (string): column in "data" to hue on
        ci (str, optional): Confidence interval for bars. Defaults to "sd".
        palette (str, optional): Plot Aesthetic. Defaults to "dark".
        alpha (float, optional): Transparency of the bar colors. Defaults to 0.6.
        height (int, optional): Height of the bars. Defaults to 10.
        ylim (tuple, optional): Start and end for y-axis of plot. Defaults to (0.7, 1).
    """
    g = sns.catplot(
        data=data, kind="bar",
        x="model_name", y=metric, hue=hue,
        ci=ci, palette=palette, alpha=alpha, height=height
    )
    g.set(ylim=ylim)