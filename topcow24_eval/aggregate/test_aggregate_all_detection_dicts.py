import pandas as pd
from aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
    count_grp2_detection,
    get_dect_avg,
)


def test_count_grp2_detection():
    # reuse detection_dict from seg_metrics/test_detection_grp2_labels.py

    # List of dictionaries
    dicts = [
        {
            "8": {"label": "R-Pcom", "Detection": "TN"},
            "9": {"label": "L-Pcom", "Detection": "TN"},
            "10": {"label": "Acom", "Detection": "TP"},
            "15": {"label": "3rd-A2", "Detection": "TP"},
        },
        {
            # label-8 IoU = 0.25 -> TP
            "8": {"label": "R-Pcom", "Detection": "TP"},
            # label-9 IoU < 0.25 -> FN
            "9": {"label": "L-Pcom", "Detection": "FN"},
            # label-10 IoU > 0.25 -> TP
            "10": {"label": "Acom", "Detection": "TP"},
            # label-15 GT missing, pred not -> FP
            "15": {"label": "3rd-A2", "Detection": "FP"},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_detection_dicts = pd.Series(dicts)

    assert count_grp2_detection(all_detection_dicts) == {
        "8": {"label": "R-Pcom", "TN": 1, "TP": 1, "FN": 0, "FP": 0},
        "9": {"label": "L-Pcom", "TN": 1, "TP": 0, "FN": 1, "FP": 0},
        "10": {"label": "Acom", "TN": 0, "TP": 2, "FN": 0, "FP": 0},
        "15": {"label": "3rd-A2", "TN": 0, "TP": 1, "FN": 0, "FP": 1},
    }


def test_get_dect_avg():
    # counts from test_count_grp2_detection()
    detection_counts = {
        "8": {"label": "R-Pcom", "TN": 1, "TP": 1, "FN": 0, "FP": 0},
        "9": {"label": "L-Pcom", "TN": 1, "TP": 0, "FN": 1, "FP": 0},
        "10": {"label": "Acom", "TN": 0, "TP": 2, "FN": 0, "FP": 0},
        "15": {"label": "3rd-A2", "TN": 0, "TP": 1, "FN": 0, "FP": 1},
    }

    assert get_dect_avg(detection_counts) == {
        "8": {"label": "R-Pcom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "9": {"label": "L-Pcom", "precision": 0, "recall": 0.0, "f1_score": 0.0},
        "10": {"label": "Acom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "15": {
            "label": "3rd-A2",
            "precision": 0.5,
            "recall": 1.0,
            "f1_score": 0.6666666666666666,
        },
        "precision": {"mean": 0.625, "std": 0.414578098794425},
        "recall": {"mean": 0.75, "std": 0.4330127018922193},
        "f1_score": {"mean": 0.6666666666666666, "std": 0.408248290463863},
    }

    # counts from
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall
    detection_counts = {
        # A model outputs 5 TP, 6 TN, 3 FP, and 2 FN. Calculate the recall.
        "8": {"label": "R-Pcom", "TP": 5, "TN": 6, "FP": 3, "FN": 2},
        # A model outputs 3 TP, 4 TN, 2 FP, and 1 FN. Calculate the precision.
        "9": {"label": "L-Pcom", "TP": 3, "TN": 4, "FP": 2, "FN": 1},
        # Precision 0.85, Recall 0.83
        "10": {"label": "Acom", "TN": 44, "TP": 40, "FN": 8, "FP": 7},
        # Precision 0.97, Recall 0.63
        "15": {"label": "3rd-A2", "TN": 50, "TP": 30, "FN": 18, "FP": 1},
    }
    assert get_dect_avg(detection_counts) == {
        # Recall is calculated as [\frac{TP}{TP+FN}=\frac{5}{7}]. = 0.714
        "8": {
            "label": "R-Pcom",
            "precision": 0.625,
            "recall": 0.7142857142857143,
            "f1_score": 0.6666666666666666,
        },
        # Precision is calculated as [\frac{TP}{TP+FP}=\frac{3}{5}]. = 0.6
        "9": {
            "label": "L-Pcom",
            "precision": 0.6,
            "recall": 0.75,
            "f1_score": 0.6666666666666666,
        },
        # Precision 0.85, Recall 0.83
        "10": {
            "label": "Acom",
            "precision": 0.851063829787234,
            "recall": 0.8333333333333334,
            "f1_score": 0.8421052631578947,
        },
        # Precision 0.97, Recall 0.63
        "15": {
            "label": "3rd-A2",
            "precision": 0.967741935483871,
            "recall": 0.625,
            "f1_score": 0.759493670886076,
        },
        "precision": {"mean": 0.7609514413177763, "std": 0.15432977020958924},
        "recall": {"mean": 0.730654761904762, "std": 0.0747462402075855},
        "f1_score": {"mean": 0.7337330668443259, "std": 0.07315043697750956},
    }


def test_aggregate_all_detection_dicts():
    """
    combine test_count_grp2_detection()
    and test_get_dect_avg()
    """
    # reuse detection_dict from seg_metrics/test_detection_grp2_labels.py

    # List of dictionaries
    dicts = [
        {
            "8": {"label": "R-Pcom", "Detection": "TN"},
            "9": {"label": "L-Pcom", "Detection": "TN"},
            "10": {"label": "Acom", "Detection": "TP"},
            "15": {"label": "3rd-A2", "Detection": "TP"},
        },
        {
            # label-8 IoU = 0.25 -> TP
            "8": {"label": "R-Pcom", "Detection": "TP"},
            # label-9 IoU < 0.25 -> FN
            "9": {"label": "L-Pcom", "Detection": "FN"},
            # label-10 IoU > 0.25 -> TP
            "10": {"label": "Acom", "Detection": "TP"},
            # label-15 GT missing, pred not -> FP
            "15": {"label": "3rd-A2", "Detection": "FP"},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_detection_dicts = pd.Series(dicts)

    assert aggregate_all_detection_dicts(all_detection_dicts) == {
        "8": {"label": "R-Pcom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "9": {"label": "L-Pcom", "precision": 0, "recall": 0.0, "f1_score": 0.0},
        "10": {"label": "Acom", "precision": 1.0, "recall": 1.0, "f1_score": 1.0},
        "15": {
            "label": "3rd-A2",
            "precision": 0.5,
            "recall": 1.0,
            "f1_score": 0.6666666666666666,
        },
        "precision": {"mean": 0.625, "std": 0.414578098794425},
        "recall": {"mean": 0.75, "std": 0.4330127018922193},
        "f1_score": {"mean": 0.6666666666666666, "std": 0.408248290463863},
    }
