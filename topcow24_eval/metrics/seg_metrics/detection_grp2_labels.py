"""
for Group 2 CoW components average F1 score
(F1 calculated during aggregate stage)

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from typing import Dict

import SimpleITK as sitk
from topcow24_eval.constants import (
    DETECTION,
    GROUP2_COW_COMPONENTS_LABELS,
    IOU_THRESHOLD,
    MUL_CLASS_LABEL_MAP,
)
from topcow24_eval.utils.utils_mask import extract_labels


def iou_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> float:
    """use overlap measures filter with a single label"""
    print(f"\n\tfor label-{label}")
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    # Jaccard Coefficient is IoU
    iou_score = overlap_measures.GetJaccardCoefficient(label)
    print("\tiou_score = ", iou_score)
    return iou_score


def detection_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> str:
    """
    run detection for a single label

    Detection can be
    TP (true positive):
        (label is present in GT) ^ (IoU >= Threshold)
    TN (true negative):
        (label is absent in GT) ^ (absent in Pred)
    FP (false positive):
        (label is absent in GT) ^ (present in Pred)
    FN (false negative):
        (label is present in GT) ^ (IoU < Threshold)

    Returns
        detection string based on DETECTION enum (TP/TN/FP/FN)
    """

    gt_array = sitk.GetArrayFromImage(gt)
    pred_array = sitk.GetArrayFromImage(pred)

    gt_labels = extract_labels(gt_array)
    pred_labels = extract_labels(pred_array)

    # check if label is present in GT
    if label in gt_labels:
        print("\tlabel is present in GT")

        # now decide between TP and FN based on IoU
        iou_score = iou_single_label(gt=gt, pred=pred, label=label)

        if iou_score >= IOU_THRESHOLD:
            # TP (true positive):
            # (label is present in GT) ^ (IoU >= Threshold)
            detection = DETECTION.TP.value
        else:
            # FN (false negative):
            # (label is present in GT) ^ (IoU < Threshold)
            detection = DETECTION.FN.value
    else:
        print("\tlabel is absent in GT")

        # now decide between FP and TN based on pred_labels
        if label in pred_labels:
            print("\tlabel is present in Pred")
            # FP (false positive):
            # (label is absent in GT) ^ (present in Pred)
            detection = DETECTION.FP.value
        else:
            print("\tlabel is absent in Pred")
            # TN (true negative):
            # (label is absent in GT) ^ (absent in Pred)
            detection = DETECTION.TN.value

    print(f"\tdetection = {detection}")

    return detection


def detection_grp2_labels(*, gt: sitk.Image, pred: sitk.Image) -> Dict:
    """
    for Group 2 CoW component labels
    regardless of presence in gt or pred,
    calculate the detection and returns
    a dict with str based on DETECTION enum (TP/TN/FP/FN)

    Returns
        detection_dict
        e.g.
        {
            "8": {"label": "R-Pcom", "Detection": "TP"}
            "9": {"label": "L-Pcom", "Detection": "FP"}
            "10": {"label": "Acom", "Detection": "FN"}
            "15": {"label": "3rd-A2", "Detection": "TN"}
        }
    """
    print("\nDetection of Grp2 Labels >>>")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    detection_dict = {}

    # irregardless of presence, go through each Grp2 Cow label
    for label in GROUP2_COW_COMPONENTS_LABELS:
        print(f"\nGrp2 label-{label}\n")

        detection = detection_single_label(gt=gt, pred=pred, label=label)

        # populate the detection result based on label's key
        detection_dict[str(label)] = {
            "label": MUL_CLASS_LABEL_MAP[str(label)],
            "Detection": detection,
        }

    print("\ndetection_grp2_labels() =>")
    pprint.pprint(detection_dict, sort_dicts=False)
    return detection_dict
