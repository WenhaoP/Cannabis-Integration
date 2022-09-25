####################################################
## Author: Kendall Kikkawa
## Institution: Berkeley Institute for Data Science
## Date: Summer 2021, Fall 2021
####################################################

"""
General framework for training models
"""

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import itertools as it

from utils import *
from params import *
from evaluate import *

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, dilation_rate=1):
    """[summary]

    Args:
        num_filters ([type]): the number of filters per filter size
        kernel_size ([type]): the number of words we want our convolutional filters to cover. 
                              We will have num_filters for each size specified here. 
            - For example, [2, 3] means that we will have filters that slide over 2 and 3 words, respectively.
        vocab_size ([type]): the size of our vocabulary. 
            - This is needed to define the size of our embedding layer, which will have shape [vocabulary_size, embedding_size]
        embedding_dim ([type]): the dimensionality of our embeddings
        maxlen ([type]): maximum number of words kept in the product description
        dilation_rate (int, optional): skip size of the convolution on the embedding. Defaults to 1.

    Returns:
        [type]: [description]
    """
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters[2], kernel_size, activation='relu', dilation_rate=dilation_rate))
    model.add(layers.Conv1D(num_filters[1], kernel_size, activation='relu'))
    model.add(layers.Conv1D(num_filters[0], kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train(insample, outsample, stop_words=True):
    best_min_f1 = 0
    best_hyperparameters = []

    descriptions = insample['straindescription']
    descriptions_test = outsample['straindescription']

    # Iterate over all hyperparanmeter combinations
    for hyperparameter_combination in SAMPLE_COMBINATIONS:
        cols_val = {colname: [] for colname in COLUMN_NAMES}
        cols_test = {colname: [] for colname in COLUMN_NAMES}

        num_filters = hyperparameter_combination[0]
        kernel_size = hyperparameter_combination[1]
        dilation_rate = hyperparameter_combination[2]
        vocab_size = hyperparameter_combination[3]
        embedding_dim = hyperparameter_combination[4]
        maxlen = hyperparameter_combination[5]
        
        # Train for each unique label
        for label in LABELS + BALANCED_LABELS:
            balanced_f = 'undersampled' in label

            if balanced_f:
                label = label.split('_')[0]
            Y = insample[label]

            # Train-test split
            descriptions_train, descriptions_val, y_train, y_val = train_test_split(
                descriptions, Y, test_size = VALIDATION_SET_SIZE, random_state = SEED)
            
            if balanced_f:
                # Balancing the class distribution
                class_0_f = (y_train == 0)
                class_0_n = class_0_f.sum()
                class_0_descriptions = descriptions_train[class_0_f].reset_index(drop=True)

                class_1_f = (y_train == 1)
                class_1_n = class_1_f.sum()
                class_1_descriptions = descriptions_train[class_1_f].reset_index(drop=True)

                if class_0_n > class_1_n:
                    class_0_descriptions = class_0_descriptions.sample(class_1_n, random_state=SEED)
                else:
                    class_1_descriptions = class_1_descriptions.sample(class_0_n, random_state=SEED)

                descriptions_train = pd.concat([class_0_descriptions, class_1_descriptions], axis=0)
                y_train = pd.Series(np.concatenate([np.zeros(len(class_0_descriptions), dtype=int), np.ones(len(class_1_descriptions), dtype=int)]))
                y_test = outsample[label]
                label = label + '_undersampled'
            else:
                y_test = outsample[label]

            # Tokenize words
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(descriptions_train)
            X_train = tokenizer.texts_to_sequences(descriptions_train)
            X_val = tokenizer.texts_to_sequences(descriptions_val)
            X_test = tokenizer.texts_to_sequences(descriptions_test)
            
            # Adding 1 because of reserved 0 index
            vocab_size = len(tokenizer.word_index) + 1

            # Pad sequences with zeros
            X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
            X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)  
            X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)  

            model = create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, dilation_rate)

            EARLY_STOPPING = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=2)
            model.fit(X_train, y_train,
                        epochs = NUM_EPOCHS,
                        verbose = VERBOSE,
                        batch_size = BATCH_SIZE,
                        validation_data = (X_val, y_val),
                        callbacks = [EARLY_STOPPING])
            
            # Evaluate validation set
            metrics = get_metrics(X_val, y_val, model, num_filters, kernel_size, dilation_rate, vocab_size, 
                                    embedding_dim, maxlen, VALIDATION_SET_SIZE)
            # Append to validation metrics file
            cols_val['model_name'].append(label)
            for metric, val in metrics.items():
                cols_val[metric].append(val)
                
            # Evaluate testing set
            metrics = get_metrics(X_test, y_test, model, num_filters, kernel_size, dilation_rate, vocab_size, 
                                    embedding_dim, maxlen, VALIDATION_SET_SIZE)
            # Append to test metrics file
            cols_test['model_name'].append(label)
            for metric, val in metrics.items():
                cols_test[metric].append(val)

        traintestsplit = '{}.{}'.format(int(100 - VALIDATION_SET_SIZE*100), int(VALIDATION_SET_SIZE*100))
        
        output = pd.DataFrame(cols_val)
        output = output[COLUMN_NAMES]
        output.to_csv('./metrics/validation/metrics_traintestsplit({})_numfilters({})_kernelsize({})_dilation({})_vocab_size({})_embeddingdim({})_maxlen({}).csv'.format(
            traintestsplit, num_filters, kernel_size, dilation_rate, vocab_size, embedding_dim, maxlen), 
            index=False, quoting=csv.QUOTE_NONNUMERIC)
        
        output = pd.DataFrame(cols_test)
        output = output[COLUMN_NAMES]
        output.to_csv('./metrics/test/metrics_traintestsplit({})_numfilters({})_kernelsize({})_dilation({})_vocab_size({})_embeddingdim({})_maxlen({}).csv'.format(
            traintestsplit, num_filters, kernel_size, dilation_rate, vocab_size, embedding_dim, maxlen), 
            index=False, quoting=csv.QUOTE_NONNUMERIC)
