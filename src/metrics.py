import numpy as np
from typing import Tuple, List, Union

def compute_confusion_counts(
    y_true: Union[List[int], np.ndarray], 
    y_pred: Union[List[int], np.ndarray]
) -> Tuple[int, int, int, int]:
    """
    Computes the counts of TP, FP, TN, FN.
    
    Args:
        y_true: Ground truth labels (1 for same person, 0 for different).
        y_pred: Predicted labels (1 for predicted same, 0 for predicted different).
        
    Returns:
        Tuple containing (TP, FP, TN, FN) as integers.
    """
    # Convert to numpy arrays just in case lists are passed in
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Check that lengths match!
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    # True Positives: Actual is 1 AND Predicted is 1
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # False Positives: Actual is 0 BUT Predicted is 1
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # True Negatives: Actual is 0 AND Predicted is 0
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # False Negatives: Actual is 1 BUT Predicted is 0
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Convert numpy int to standard python int for JSON serialization later
    return int(tp), int(fp), int(tn), int(fn)

def compute_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    total = tp + fp + tn + fn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    # Precision: Out of all predicted matches, how many were correct?
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: Out of all actual matches, how many did we find?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1: Harmonic mean of Precision and Recall
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def compute_balanced_accuracy(tp: int, fp: int, tn: int, fn: int) -> float:
    """
    Computes balanced accuracy: (True Positive Rate + True Negative Rate) / 2
    """
    # True Positive Rate (Sensitivity / Recall)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # True Negative Rate (Specificity)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return (tpr + tnr) / 2.0


def summarize_metrics(y_true, y_pred) -> dict:
    """
    Wrapper function that computes all metrics and returns them as a dictionary.
    Used for saving directly to a JSON file for experiment tracking.
    """
    # 1. Get the base counts
    tp, fp, tn, fn = compute_confusion_counts(y_true, y_pred)
    
    # 2. Compute all the derived metrics
    accuracy = compute_accuracy(tp, fp, tn, fn)
    precision, recall, f1 = compute_precision_recall_f1(tp, fp, fn)
    balanced_acc = compute_balanced_accuracy(tp, fp, tn, fn)
    
    # 3. Pack them into a neat, JSON-serializable dictionary
    return {
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn
        },
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
