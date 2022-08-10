####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Summer 2021, Fall 2021
####################################################

"""
Contains pipeline and modeling parameters that should be fixed at run time, but that a user may want to pre-specify
to compare various results
"""

from re import VERBOSE
import sklearn
from sklearn.model_selection import ParameterGrid
import itertools as it
import random

# Reproducibility
SEED = 1000

# Keep the stop words or not
STOP_WORDS = True

#validation set size
VALIDATION_SET_SIZE = 0.2

# Class labels for model prediction
# LABELS = ['Cannabinoid', 'Genetics', 'Intoxication', 'Look', 'Medical', 'Smell Flavor', 'Wellness', 'Commoditization']
LABELS = ['Cannabinoid', 'Intoxication', 'Medical', 'Wellness', 'Commoditization']

# Hyperparameter Grid
HYPERPARAMETER_GRID = {
    "num_filters": [[16, 32, 64], [32, 64, 128], [64, 128, 256]],
    "kernel_size": [2, 3, 4], 
    "dilation": [1, 2],
    "vocab_size": [5000], 
    "embedding_dim": [16, 32, 64],
    "maxlen": [100, 120, 140]}

ALL_COMBINATIONS = it.product(*(HYPERPARAMETER_GRID[key] for key in HYPERPARAMETER_GRID))
# ALL_COMBINATIONS = ParameterGrid((HYPERPARAMETER_GRID))
ALL_COMBINATIONS = list(ALL_COMBINATIONS)

# Define # of combinations of hyperparameters to consider
NUM_COMBINATIONS = 5
assert(NUM_COMBINATIONS <= len(ALL_COMBINATIONS))

random.seed(SEED)
SAMPLE_COMBINATIONS = random.sample(ALL_COMBINATIONS, k=NUM_COMBINATIONS)

# Columns in Insample, Outsample
COLUMN_NAMES = ['model_name', 'pred_true', 'pred_false', 'actual_true', 'actual_false', 'accuracy',\
                'precision', 'recall', 'f1', 'sensitivity', 'specificity', 'informedness',\
                'matthews_correlation', 'cohen_kappa', 'roc_auc', 'tn', 'fp', 'fn', 'tp', 
                'num_filters', 'kernel_size', 'dilation', 'vocab_size', 'embedding_dim', 'maxlen', 'validation_set_size']

# Training Parameters
NUM_EPOCHS = 20
BATCH_SIZE = 256


# Validation Results: change to display more/less best models
NUM_BEST_MODELS = 1

# Print out training process
VERBOSE = False

