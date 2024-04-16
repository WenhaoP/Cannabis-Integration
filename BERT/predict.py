import pandas as pd
import numpy as np
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from datasets import Dataset
from torch.nn import functional as F
from torch import from_numpy

from params import *
from utils import clean_data, preprocess_function

def prediction(down_sample=False):
    full_dataset = pd.read_csv("data/full_dataset.csv")
    if down_sample:
        full_dataset = full_dataset.sample(50000, random_state=RANDOM_STATE)
    
    full_dataset['straindescription'] = '"' + full_dataset['strain'].astype(str) + '" -- '+ full_dataset['description'].astype(str)
    clean_full = clean_data(full_dataset, "straindescription", [], punctuations=PUNCTUATIONS, stop_words=STOP_WORDS, digits=DIGITS, minimal=True)

    # preprocess the textual input 
    dataset = Dataset.from_pandas(clean_full)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    # tokenized_dataset = tokenized_dataset.remove_columns(["straindescription", "__index_level_0__"])
    tokenized_dataset = tokenized_dataset.remove_columns(["straindescription"])

    for label in LABELS:

        print(f'Predicting {label}...')

        # set up directory paths
        best_model_dir = "best_bert_" + label

        model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)

        trainer = Trainer(
            model=model,
        )

        predictions = trainer.predict(tokenized_dataset)
        predict_labels = np.argmax(predictions.predictions, axis=-1)
        full_dataset[(label+"_labeled").lower()] = predict_labels
        full_dataset[(label+"_logit").lower()] = np.max(predictions.predictions, axis=-1)
        full_dataset[(label+"_prob").lower()] = F.softmax(from_numpy(predictions.predictions), dim=-1).numpy()[:,-1]

    # manipulate the dataframe so that it is acceptable to the washington pipeline code
    # full_dataset = full_dataset.rename({"Medical_labeled":"Medical_undersampled_labeled",
    #  "medical_labeled":"medical_undersampled_labeled"}, axis=1)

    for unpredict_label in (set(FULL_LABELS) - set(LABELS)):
        full_dataset[f"{unpredict_label}_labeled"] = np.ones(full_dataset.shape[0]).astype(int) * -1
        full_dataset[f"{unpredict_label}_logit"] = np.ones(full_dataset.shape[0]).astype(int) * -1
        full_dataset[f"{unpredict_label}_prob"] = np.ones(full_dataset.shape[0]).astype(int) * -1
    full_dataset = full_dataset.rename({"Medical_labeled":"medical_labeled"}, axis=1)
    full_dataset["medical_undersampled_labeled"] = np.ones(full_dataset.shape[0]).astype(int) * -1
    full_dataset["medical_undersampled_logit"] = np.ones(full_dataset.shape[0]).astype(int) * -1
    full_dataset["medical_undersampled_prob"] = np.ones(full_dataset.shape[0]).astype(int) * -1

    full_dataset.to_csv("data/full_dataset_with_labels.csv", index=False, line_terminator='\r\n')