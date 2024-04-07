import shutil
import pandas as pd
import os
import numpy as np
import transformers
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import train_test_split 

from params import *
from utils import load_data, preprocess_function
from evaluate import compute_metrics

def train():

    try:
        os.remove("metrics/train_val_evaluation.csv")
        os.remove("metrics/train_test_evaluation.csv")
    except:
        None

    val_eval = {}
    test_eval = {}

    # fine-tune a separate model for each label
    for label in LABELS:
        print(f"=====Start running for the {label} label=====")

        if os.path.exists("metrics/hp_tune_test_evaluation.csv"):
            test_metric_table = pd.read_csv("metrics/hp_tune_val_evaluation.csv", index_col=0)
            test_metric_table_label = test_metric_table.loc[test_metric_table.index == label,]
            test_metric_table_label = test_metric_table_label.sort_values(['test_f1_score','learning_rate','weight_decay','batch_size'], ascending=[False, True, True, True])
            
            LR = test_metric_table_label.iloc[0]['learning_rate']
            BATCH_SIZE = int(test_metric_table_label.iloc[0]['batch_size'])
            GRAD_ACC_STEPS = (64 // BATCH_SIZE)
            WEIGHT_DECAY = test_metric_table_label.iloc[0]['weight_decay']

            print("=====Using the best hyperparameter combination from hyperparameter tuning=====")
            print(f"=====The best hyperparameter combination is: learning rate = {LR}, batch size = {BATCH_SIZE}, weight decay = {WEIGHT_DECAY}=====")
        else:
            print("=====Using the default hyperparameter combination in params.py=====")

        # load the datasets
        # raw_insample = pd.read_csv("data/in_sample.csv")
        # raw_outsample = pd.read_csv("data/out_sample.csv")
        # clean_insample, clean_outsample = load_data("straindescription", LABELS, punctuations=PUNCTUATIONS, stop_words=STOP_WORDS, minimal=True)
        # train, val = train_test_split(clean_insample, test_size=VAL_SIZE, random_state=RANDOM_STATE)
        # train.to_csv('data/train.csv', index=False)
        # val.to_csv('data/val.csv', index=False)
        dataset = load_dataset('csv', data_files={'train': ['data/train.csv'], 'val': ['data/val.csv'], 'test': ['data/clean_out_sample.csv']})

        # preprocess the textual input 
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        if (label == 'Medical_Wellness'):
            # print('HERE!!!')
            tokenized_dataset['train'] = tokenized_dataset['train'].add_column("Medical_Wellness", np.logical_or(tokenized_dataset['train']['Medical'], tokenized_dataset['train']['Wellness']).astype(int))
            tokenized_dataset['val'] = tokenized_dataset['val'].add_column("Medical_Wellness", np.logical_or(tokenized_dataset['val']['Medical'], tokenized_dataset['val']['Wellness']).astype(int))
            tokenized_dataset['test'] = tokenized_dataset['test'].add_column("Medical_Wellness", np.logical_or(tokenized_dataset['test']['Medical'], tokenized_dataset['test']['Wellness']).astype(int))
        if (label == 'Pre_Hybrid'):
            tokenized_dataset['train'] = tokenized_dataset['train'].add_column("Pre_Hybrid", np.logical_and(
                np.logical_or(tokenized_dataset['train']['Medical'], tokenized_dataset['train']['Wellness']).astype(int), 
                dataset['train']['Intoxication']).astype(int))
            tokenized_dataset['val'] = tokenized_dataset['val'].add_column("Pre_Hybrid", np.logical_and(
                np.logical_or(tokenized_dataset['val']['Medical'], tokenized_dataset['val']['Wellness']).astype(int), 
                dataset['val']['Intoxication']).astype(int))
            tokenized_dataset['test'] = tokenized_dataset['test'].add_column("Pre_Hybrid", np.logical_and(
                np.logical_or(tokenized_dataset['test']['Medical'], tokenized_dataset['test']['Wellness']).astype(int), 
                tokenized_dataset['test']['Intoxication']).astype(int))
        tokenized_dataset = tokenized_dataset.remove_columns("straindescription")

        # set up directory paths
        model_dir = 'bert_' + label
        best_model_dir = 'best_' + model_dir
        best_model_dir_zip = 'best_' + model_dir + '.zip'
        try:
            shutil.rmtree(model_dir) # remove possible cache
            shutil.rmtree(best_model_dir)
            print("remove")
            os.remove(best_model_dir_zip)
        except:
            None
        
        # remove other labels and rename the target label
        # other_labels = list(set(list(filter(lambda x: x != label, LABELS))))
        # tokenized_dataset_label = tokenized_dataset.remove_columns(other_labels)
        tokenized_dataset_label = tokenized_dataset.remove_columns(set(tokenized_dataset['train'].column_names) - set([label] + ['input_ids', 'token_type_ids', 'attention_mask']))
        tokenized_dataset_label = tokenized_dataset_label.rename_column(label, "label")

        # load the pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            classifier_dropout=CLASSIFIER_DROPOUT,
            num_labels=NUM_CLASSES,
        )

        print(f"=====The Actual Parameters is: learning rate = {LR}, batch size = {BATCH_SIZE}, weight decay = {WEIGHT_DECAY}=====")
        # set up the training arguments
        training_args = TrainingArguments(
            output_dir=model_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY, 
            num_train_epochs=EPOCHS,
            lr_scheduler_type=LR_SCHEDULER,
            evaluation_strategy=STRATEGY,
            logging_strategy=STRATEGY, 
            save_strategy=STRATEGY,
            eval_steps=STEPS,
            logging_steps=STEPS,
            save_steps=STEPS,
            seed=SEED,
            load_best_model_at_end=True,
            metric_for_best_model=EVAL_METRIC,
            report_to="none"
        )

        # set up the trainer 
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_label['train'],
            eval_dataset=tokenized_dataset_label['val'],
            tokenizer=TOKENIZER,   
            compute_metrics=compute_metrics,
        )

    #     # train (fine-tune) the model
    #     trainer.train()

    #     # evaluate the best model
    #     val_predictions = trainer.predict(tokenized_dataset_label["val"])
    #     val_eval[label] = val_predictions.metrics
    #     test_predictions = trainer.predict(tokenized_dataset_label["test"])
    #     test_eval[label] = test_predictions.metrics

    #     # save the best model
    #     model.save_pretrained(best_model_dir)
    #     # os.system(f'zip -r {best_model_dir_zip} {best_model_dir}')
    #     shutil.rmtree(model_dir)
    #     # shutil.rmtree(best_model_dir)

    # # save the evaluation result of each model
    # val_eval_df = pd.DataFrame.from_dict(val_eval).transpose()
    # val_eval_df.to_csv(f"metrics/train_val_evaluation.csv")
    # test_eval_df = pd.DataFrame.from_dict(test_eval).transpose()
    # test_eval_df.to_csv(f"metrics/train_test_evaluation.csv")