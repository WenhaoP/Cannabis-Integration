import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv

from params import *
from train import create_model
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

def prediction(insample, full_dataset, best_val_models):
    """
    Use the best models on the validation set to predict
    """
    descriptions = insample['straindescription']
    descriptions_test = full_dataset['straindescription']
    preds_dict = {}

    for index, row in best_val_models.iterrows():
        label = row['model_name']
        balanced_f = 'undersampled' in label
        num_filters = row['num_filters'].strip('][').split(', ')
        num_filters = [int(i) for i in num_filters]
        kernel_size = int(row['kernel_size'])
        dilation_rate = int(row['dilation'])
        vocab_size = int(row['vocab_size'])
        embedding_dim = int(row['embedding_dim'])
        maxlen = int(row['maxlen'])
        vs = float(row['validation_set_size'])

        print(f'=== Predicting {label} ===')    

        if balanced_f:
            label = label.split('_')[0]
        Y = insample[label]

        # Train-test split
        descriptions_train, descriptions_val, y_train, y_val = train_test_split(
            descriptions, Y, test_size=vs, random_state=1000)
        
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
            label = label + '_undersampled'

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

        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=2)
        model.fit(X_train, y_train,
                    epochs=20,
                    verbose=VERBOSE,
                    batch_size=256,
                    validation_data=(X_val, y_val),
                    callbacks=[es])

        # Evaluate testing set
        predictions = model.predict(X_test)
        predictions = predictions.round()
        
        preds_dict[label] = predictions
        print(f'Positive rate: {np.mean(predictions)}')

    for label in preds_dict:
        full_dataset[f'{label}_labeled'] = preds_dict[label] 
    
    for unpredict_label in (set(FULL_LABELS) - set(LABELS)):
        full_dataset[f"{unpredict_label}_labeled"] = np.zeros(full_dataset.shape[0]).astype(int)

    return full_dataset 