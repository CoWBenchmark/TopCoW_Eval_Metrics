"""
Class-average Dice similarity coefficient

Metrics for Task-1-CoW-Segmentation
"""

import pprint

import numpy as np
import SimpleITK as sitk
from topcow24_eval.metrics.seg_metrics.generate_cls_avg_dict import (
    generate_cls_avg_dict,
)


def dice_coefficient_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> float:
    """
    use overlap measures filter with a single label
    for Dice Similarity Coefficient (DSC)

    NOTE: DSC in sitk.LabelOverlapMeasuresImageFilter
    does NOT use voxel spacing. It calculates the count
    of voxels without considering the area/volume.

    NOTE: two sitk.Images with DIFFERENT voxel spacings
    can still be calculated for overlap measures...(?!)
    """
    print(f"\nfor label-{label}")

    # Check if label exists for both gt and pred
    # If not, DSC is automatically set to 0 due to FP or FN
    gt_label_arr = sitk.GetArrayFromImage(gt == label)
    pred_label_arr = sitk.GetArrayFromImage(pred == label)

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        print(f"[!!Warning] label-{label} empty for gt or pred")
        return 0

    # NOTE: sometimes there are tiny differences in image Direction:
    # ITK ERROR: LabelOverlapMeasuresImageFilter(0x55bf9ad3e5e0):
    # Inputs do not occupy the same physical space!
    # Thus make sure they have the same metadata
    # Copies the Origin, Spacing, and Direction from the gt image
    pred.CopyInformation(gt)

    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    dice_score = overlap_measures.GetDiceCoefficient(label)
    print("dice_score = ", dice_score)
    return dice_score


def dice_coefficient_all_classes(*, gt: sitk.Image, pred: sitk.Image) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with dice_coefficient_single_label() as metric_func
    """
    dice_dict = generate_cls_avg_dict(
        gt=gt,
        pred=pred,
        metric_keys=["Dice"],
        metric_func=dice_coefficient_single_label,
    )
    print("\ndice_coefficient_all_classes() =>")
    pprint.pprint(dice_dict, sort_dicts=False)
    return dice_dict
