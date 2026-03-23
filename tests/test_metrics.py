import pytest
import sys
from pathlib import Path
import numpy as np
from src.metrics import compute_confusion_counts, compute_accuracy, compute_precision_recall_f1, compute_balanced_accuracy, summarize_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

Y_TRUE = [1, 0, 1, 0, 1]
Y_PRED = [1, 0, 0, 1, 1]

def test_compute_confusion_counts():
    tp, fp, tn, fn = compute_confusion_counts(Y_TRUE, Y_PRED)
    assert tp == 2
    assert fp == 1
    assert tn == 1
    assert fn == 1

def test_compute_accuracy():
    # Using our known counts: tp=2, fp=1, tn=1, fn=1 (Total = 5)
    # Correct (TP + TN) = 3. Total = 5. Accuracy should be 0.6.
    acc = compute_accuracy(2, 1, 1, 1)
    assert acc == 0.6

def test_compute_precision_recall_f1():
    precision, recall, f1 = compute_precision_recall_f1(2, 1, 1)
    # Precision: 2 / (2 + 1) = 0.666...
    # Recall: 2 / (2 + 1) = 0.666...
    assert pytest.approx(precision, 0.01) == 0.666
    assert pytest.approx(recall, 0.01) == 0.666
    assert pytest.approx(f1, 0.01) == 0.666

def test_compute_balanced_accuracy():
    # TPR = 2/3. TNR = 1/2.
    # Balanced Acc = (0.666... + 0.5) / 2 = 0.58333...
    b_acc = compute_balanced_accuracy(2, 1, 1, 1)
    assert pytest.approx(b_acc, 0.01) == 0.583

def test_summarize_metrics():
    # Test that the wrapper function correctly builds the dictionary
    results = summarize_metrics(Y_TRUE, Y_PRED)
    
    assert "confusion_matrix" in results
    assert results["confusion_matrix"]["TP"] == 2
    assert results["accuracy"] == 0.6
    assert "balanced_accuracy" in results
    assert "f1_score" in results