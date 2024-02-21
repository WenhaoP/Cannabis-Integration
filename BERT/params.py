####################################################
## Author: Wenhao Pan
## Institution: Berkeley Institute for Data Science
## Spring 2022, Summer 2022
####################################################

"""
Contains pipeline and modeling parameters that should be fixed at run time, but that a user may want to pre-specify
to compare various results
"""
from transformers import AutoTokenizer

# the name of the pre-trained model we want to use
MODEL_NAME = "bert-base-uncased" 

FULL_LABELS = ['Cannabinoid', 'Genetics', 'Intoxication', 'Look', 'Medical', 'Smell Flavor', 'Wellness', 'Commoditization']
LABELS = ["Intoxication", "Medical_Wellness", "Commoditization"] # Medical_Wellness = 1 if (Medical == 1) OR (Wellness = 1)

### Preprocess Setup ###
# text cleaning hyperparameters
PUNCTUATIONS = True
STOP_WORDS = True
DIGITS = True
MINIMAL = True

# dataset splitting hyperparameters
VAL_SIZE = 0.2 # validation set size
RANDOM_STATE = 10 # random seed 

# tokenization hyperparameters
PADDING = 'max_length' # padding strategy
PADDING_SIDE = 'right' # the side on which the model should have padding applied
TRUNCATION = True # truncate strategy
TRUNCATION_SIDE = 'right' # the side on which the model should have truncation applied
MAX_LEN = 150 # maximum length to use by one of the truncation/padding parameters

# Load the pre-trained tokenmizer ###
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    padding_side=PADDING_SIDE,
    truncation_side=TRUNCATION_SIDE,
)

### Training and Model Setup ###
TRAIN = False # train new models if True

# model hyperparameters
CLASSIFIER_DROPOUT = 0.15 # dropout ratio for the classification head
NUM_CLASSES = 2 # number of classes

# optimization hyperparameters ###
SEED = 42 # random seed for splitting the data into batches
BATCH_SIZE = 16 # batch size for both training and evaluation
GRAD_ACC_STEPS = 4 # number of steps for gradient accumulation
LR = 5e-5 # initial learning rate
WEIGHT_DECAY = 2e-3 # weight decay to apply in the AdamW optimizer
EPOCHS = 8 # total number of training epochs 
LR_SCHEDULER = "cosine" # type of learning rate scheduler
STRATEGY = "steps" # strategy for logging, evaluation, and saving
STEPS = 100 # number of steps for logging, evaluation, and saving
EVAL_METRIC = "f1_score" # metric for selecting the best model

### Prediction ###
DOWN_SAMPLING = False # whether we downsample the data for prediction