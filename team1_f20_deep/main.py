from train import *
from evaluate import *
from predict import *
from params import *
from utils import *
import os
import seaborn as sns
import argparse

if __name__ == "__main__":
    tf.random.set_seed(SEED)

    if 'team1_f20_deep' not in os.getcwd():
        os.chdir(os.getcwd() + '/team1_f20_deep')

    if STOP_WORDS:
        folder = 'with_stop_words'
    else:
        folder = 'without_stop_words'

    ### Train, validate, and test models with hyperparameter tuning ###
    insample, outsample = load_data(STOP_WORDS)
    print("=== Loaded the data ===")
    train(insample, outsample)
    print("=== Finished searching the best model hyperparameters ===")
    
    best_val_models = find_best_model('validation', stop_words=STOP_WORDS)
    best_test_models = find_best_model('test', stop_words=STOP_WORDS)

    best_val_models.to_csv(f'./metrics/best_val_models_{folder}.csv', index=False)
    best_val_models[['model_name' , 'num_filters', 'kernel_size', 
                     'dilation', 'vocab_size', 'embedding_dim', 
                     'maxlen', 'validation_set_size']].to_csv(f'./metrics/best_val_model_hyperparameters_{folder}.csv', index=False)
    
    best_test_models.to_csv(f'./metrics/best_test_models_{folder}.csv', index=False)
    best_test_models[['model_name' , 'num_filters', 'kernel_size', 
                     'dilation', 'vocab_size', 'embedding_dim', 
                     'maxlen', 'validation_set_size']].to_csv(f'./metrics/best_test_model_hyperparameters_{folder}.csv', index=False)
    
    ### Plot the f1 scores of each label model ### 
    g = sns.catplot(
        data=best_val_models, kind="bar",
        x="model_name", y="f1", hue="rank",
        ci="sd", palette="dark", alpha=.6, height=10
    )
    g.set(ylim=(0.7, 1))
    g.savefig('images/val_f1_scores.png')
    
    g = sns.catplot(
        data=best_test_models, kind="bar",
        x="model_name", y="f1", hue="rank",
        ci="sd", palette="dark", alpha=.6, height=10
    )
    g.set(ylim=(0.7, 1))
    g.savefig('images/test_f1_scores.png')
    print("=== Plotted the f1 scores of each label model ===")

    ### Prediction on the full dataset ###
    full_dataset = pd.read_csv("data/full_dataset.csv")
    full_dataset['straindescription'] = '"' + full_dataset['strain'].astype(str) + '" -- '+ full_dataset['description'].astype(str)
    full_dataset = full_dataset.dropna(subset=['straindescription'])
    if not STOP_WORDS:
        full_dataset['straindescription'] = full_dataset['straindescription'].apply(remove_stopwords)
    
    full_dataset = prediction(insample, full_dataset, best_val_models)

    full_dataset.to_csv('data/full_dataset_with_labels.csv', index=False, line_terminator='\r\n')
    print("=== Finished prediction ===")
