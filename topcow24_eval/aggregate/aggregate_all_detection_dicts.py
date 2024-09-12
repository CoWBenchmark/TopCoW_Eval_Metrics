"""
aggregate the detection_dict
from pandas DataFrame: self._case_results
Get the Series: self._case_results["all_detection_dicts"]
detection_dict is under the column `all_detection_dicts`

to get the Average F1 score
(harmonic mean of the precision and recall)
for detection of the "Group 2 CoW components"
"""

import pprint

import numpy as np
from pandas import Series
from topcow24_eval.constants import GROUP2_COW_COMPONENTS_LABELS, MUL_CLASS_LABEL_MAP


def aggregate_all_detection_dicts(all_detection_dicts: Series) -> dict:
    print("\n[aggregate] aggregate_all_detection_dicts()\n")

    # get the detection counts
    detection_counts = count_grp2_detection(all_detection_dicts)
    # get the detection averages
    dect_avg = get_dect_avg(detection_counts)

    return dect_avg


def count_grp2_detection(all_detection_dicts: Series) -> dict:
    """
    input all_detection_dicts is a pandas.Series of detection_dicts
    return the count of TP|TN|FP|FN from all detection_dicts

    init the group 2 detection stats dictionary
    with label name and four 0 detection scores,
    then count the number for each detection category
    """
    detection_counts = {}

    for label in GROUP2_COW_COMPONENTS_LABELS:
        detection_counts[str(label)] = {
            "label": MUL_CLASS_LABEL_MAP[str(label)],
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
        }

    # get the values from the pandas Series
    for detection_dict in all_detection_dicts.values:
        # each detection_dict is a dictionary
        # type(detection_dict):  <class 'dict'>

        for label, value in detection_dict.items():
            detection_counts[label][value["Detection"]] += 1

    print("\ncount_grp2_detection =>")
    pprint.pprint(detection_counts, sort_dicts=False)

    return detection_counts


def get_dect_avg(detection_counts: dict) -> dict:
    """
    calculate the mean and std of detections for Group2
    compute for precision, recall, and f1_score
    also record down the detections for each Group2 class
    treat NaN as 0 during averaging for recall, precision, and f1

    Returns:
        dect_avg: dictionary

        {
            "8": {
                "label": "R-Pcom",
                "precision": 0.625,
                "recall": 0.7142857142857143,
                "f1_score": 0.6666666666666666,
            },
            "9": {...},
            "10": {...},
            "15": {...},
            "precision": {
                "mean": ...,
                "std": ...
            },
            "recall": {"mean": ..., "std": ...},
            "f1_score": {"mean": ..., "std": ...},
        }
    """
    dect_avg = {}

    # init the dect_avg dict with entries
    for label in GROUP2_COW_COMPONENTS_LABELS:
        dect_avg[str(label)] = {
            "label": MUL_CLASS_LABEL_MAP[str(label)],
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }

    list_recall = []
    list_precision = []
    list_f1 = []

    # get the detection_stats based on label
    for label, stats in detection_counts.items():
        print(f"\nfor label-{label} ({MUL_CLASS_LABEL_MAP[str(label)]})")
        tp = stats["TP"]
        fp = stats["FP"]
        fn = stats["FN"]

        # handle the undefined division by zero
        # NOTE: treat the nan as 0

        # Precision
        if (tp + fp) == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        print(f"precision = {precision}")
        list_precision.append(precision)
        dect_avg[label]["precision"] = precision

        # Recall
        if (tp + fn) == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        print(f"recall = {recall}")
        list_recall.append(recall)
        dect_avg[label]["recall"] = recall

        # F1
        if (tp + fp + fn) == 0:
            f1_score = 0
        else:
            f1_score = 2 * tp / ((2 * tp) + fp + fn)

        print(f"f1_score = {f1_score}")
        list_f1.append(f1_score)
        dect_avg[label]["f1_score"] = f1_score

    # end of for loop

    print(f"\nlist_precision = {list_precision}")
    dect_avg["precision"] = {
        "mean": np.mean(list_precision),
        "std": np.std(list_precision),
    }

    print(f"\nlist_recall = {list_recall}")
    dect_avg["recall"] = {
        "mean": np.mean(list_recall),
        "std": np.std(list_recall),
    }

    print(f"\nlist_f1 = {list_f1}")
    dect_avg["f1_score"] = {
        "mean": np.mean(list_f1),
        "std": np.std(list_f1),
    }

    print("\nget_dect_avg =>")
    pprint.pprint(dect_avg, sort_dicts=False)

    return dect_avg
