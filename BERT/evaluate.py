import numpy as np
from datasets import load_metric
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, confusion_matrix, balanced_accuracy_score

### Evaluation Metrics ###
metric_acc = load_metric("accuracy")
metric_f1 = load_metric("f1")
metric_precision = load_metric("precision")
metric_recall = load_metric("recall")
metric_auc = load_metric("roc_auc")

def compute_metrics(eval_pred):
    """
    Compute the metrics 
    ---
    Arguments:
    eval_pred (tuple): the predicted logits and truth labels

    Returns:
    metrics (dict{str: float}): contains the computed metrics 
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    prediction_scores = np.max(logits, axis=-1)

    pred_true = np.count_nonzero(predictions)
    pred_false = predictions.shape[0] - pred_true
    actual_true = np.count_nonzero(labels)
    actual_false = labels.shape[0] - actual_true

    acc = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
    f1 = metric_f1.compute(predictions=predictions, references=labels)['f1']
    precision = metric_precision.compute(predictions=predictions, references=labels)['precision']
    recall = metric_recall.compute(predictions=predictions, references=labels)['recall']
    roc_auc = metric_auc.compute(prediction_scores=predictions, references=labels)['roc_auc']
    matthews_correlation = matthews_corrcoef(y_true=labels, y_pred=predictions)
    cohen_kappa = cohen_kappa_score(y1=labels, y2=predictions)
    balanced_accuracy = balanced_accuracy_score(y_true=labels, y_pred=predictions)

    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    informedness = specificity + sensitivity - 1

    metrics = {
        "pred_true": pred_true,
        "pred_false": pred_false,
        "actual_true": actual_true,
        "actual_false": actual_false,
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "matthews_correlation": matthews_correlation,
        "cohen_kappa": cohen_kappa,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "informedness": informedness,
        "balanced_accuracy": balanced_accuracy
    }
    return metrics