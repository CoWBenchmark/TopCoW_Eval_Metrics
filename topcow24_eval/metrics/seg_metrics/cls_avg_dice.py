"""
Class-average Dice similarity coefficient

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from enum import Enum
from typing import Dict

import SimpleITK as sitk
from generate_cls_avg_dict import generate_cls_avg_dict


def dice_coefficient_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> float:
    """use overlap measures filter with a single label"""
    print(f"\nfor label-{label}")
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    dice_score = overlap_measures.GetDiceCoefficient(label)
    print("dice_score = ", dice_score)
    return dice_score


def dice_coefficient_all_classes(
    *, gt: sitk.Image, pred: sitk.Image, task: Enum
) -> Dict:
    """
    use the dict generator from generate_cls_avg_dict
    with dice_coefficient_single_label() as metric_func
    """
    dice_dict = generate_cls_avg_dict(
        gt=gt,
        pred=pred,
        task=task,
        metric_keys=["Dice"],
        metric_func=dice_coefficient_single_label,
    )
    print("\ndice_coefficient_all_classes() =>")
    pprint.pprint(dice_dict, sort_dicts=False)
    return dice_dict
