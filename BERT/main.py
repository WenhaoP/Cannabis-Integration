####################################################
## Author: Wenhao Pan
## Institution: Berkeley Institute for Data Science
## Spring 2022, Summer 2022
####################################################

"""
Training, evaluation, and prediction of the BERT text classification model
"""
import os
import torch
from transformers import AutoTokenizer

from train import train
from params import TRAIN, DOWN_SAMPLING
from predict import prediction

if __name__ == "__main__":
    if 'BERT' not in os.getcwd():
        os.chdir(os.getcwd() + '/BERT')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}...')

    assert os.path.exists("data/in_sample.csv") and os.path.exists("data/out_sample.csv"), "Raw dataset was not detected. You need to upload the dataset first!"
    
    if TRAIN:
        print('Training models from scratch...')
        train()
        
    prediction(down_sample=DOWN_SAMPLING)
