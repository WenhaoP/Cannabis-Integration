import pandas as pd
import os
import transformers
from transformers import  AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.model_selection import train_test_split 

from params import *
from utils import load_data, preprocess_function
from evaluate import compute_metrics

def train():
    val_eval = {}
    test_eval = {}

    # fine-tune a separate model for each label
    for label in LABELS:

        # load the datasets
        raw_insample = pd.read_csv("data/in_sample.csv")
        raw_outsample = pd.read_csv("data/out_sample.csv")
        clean_insample, clean_outsample = load_data("straindescription", LABELS, punctuations=PUNCTUATIONS, stop_words=STOP_WORDS, minimal=True)
        train, val = train_test_split(clean_insample, test_size=VAL_SIZE, random_state=RANDOM_STATE)
        train.to_csv('data/train.csv', index=False)
        val.to_csv('data/val.csv', index=False)
        dataset = load_dataset('csv', data_files={'train': ['data/train.csv'], 'val': ['data/val.csv'], 'test': ['data/clean_out_sample.csv']})

        # preprocess the textual input 
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns("straindescription")

        # set up directory paths
        model_dir = "bert_" + label
        best_model_dir = "best_" + model_dir
        best_model_dir_zip = "best_" + model_dir + ".zip"
        os.system(f'rm -rf {model_dir} {best_model_dir} {best_model_dir_zip}') # remove possible cache

        # remove other labels and rename the target label
        other_labels = list(filter(lambda x: x != label, LABELS))
        tokenized_dataset_label = tokenized_dataset.remove_columns(other_labels)
        tokenized_dataset_label = tokenized_dataset_label.rename_column(label, "label")

        # load the pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            classifier_dropout=CLASSIFIER_DROPOUT,
            num_labels=NUM_CLASSES,
        )

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

        # train (fine-tune) the model
        trainer.train()

        # evaluate the best model
        val_predictions = trainer.predict(tokenized_dataset_label["val"])
        val_eval[label] = val_predictions.metrics
        test_predictions = trainer.predict(tokenized_dataset_label["test"])
        test_eval[label] = test_predictions.metrics

        # save the best model
        model.save_pretrained(best_model_dir)
        os.system(f'zip -r {best_model_dir_zip} {best_model_dir}')

    # save the evaluation result of each model
    val_eval_df = pd.DataFrame.from_dict(val_eval).transpose()
    val_eval_df.to_csv("metrics/val_evaluation.csv")
    test_eval_df = pd.DataFrame.from_dict(test_eval).transpose()
    test_eval_df.to_csv("metrics/test_evaluation.csv")