"""
Class-average 0-th Betti number error

Metrics for Task-1-CoW-Segmentation
"""

import pprint

import numpy as np
import SimpleITK as sitk
from skimage import measure
from topcow24_eval.metrics.seg_metrics.generate_cls_avg_dict import (
    generate_cls_avg_dict,
)
from topcow24_eval.utils.utils_mask import arr_is_binary


def connected_components(img: np.array) -> tuple[int, list, list]:
    """
    identify connected components
    calculates the Betti number B0 for a binary 3D img

    Returns
        b0 (b0 number),
        props (the list of region properties),
        sizes (sorted sizes list)

    adapted from Betti number error calculations from repo:
    https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
    and paper:
        A skeletonization algorithm for gradient-based optimization
        ICCV 2023

    original code for all B0, B1 and B2 numbers prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    """

    # make sure the image is 3D (for connectivity settings)
    assert img.ndim == 3, "betti_number expects a 3D input"

    # make sure the image is binary with
    assert arr_is_binary(img), "betti_number works with binary input"

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected
    N26 = 3  # full connectivity of input.ndim is used

    # NOTE: for connected component analysis, no need to pad the image

    # get the label connected regions for foreground
    b0_labels, b0 = measure.label(
        img,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    print(f"# b0 = {b0}")

    # get the properties of the connected regions
    # by skimage.measure.regionprops() function
    props = measure.regionprops(b0_labels)
    sizes = [obj.area for obj in props]
    sizes.sort()
    print(f"sorted sizes = {sizes}")

    return int(b0), props, sizes


def betti_number_error_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> int:
    """
    integrate connected_components() to the template that metric_func expects:
        (gt, pred, label) => score/Tuple[scores]
    """
    print(f"\n--> betti_number_error_single_label() for label-{label}\n")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    # only need bool binary mask of the current label
    gt_label_img = gt == label
    pred_label_img = pred == label

    gt_label_arr = sitk.GetArrayFromImage(gt_label_img)
    pred_label_arr = sitk.GetArrayFromImage(pred_label_img)

    # make sure the masks are binary
    assert arr_is_binary(gt_label_arr), "expects binary gt_arr"
    assert arr_is_binary(pred_label_arr), "expects binary pred_arr"

    # if filtered label_arr is blank, b0 = 0
    print("~~~ gt_b0 ~~~")
    if not np.any(gt_label_arr):
        print("blank")
        gt_b0 = 0
    else:
        gt_b0 = connected_components(gt_label_arr)[0]

    print("~~~ pred_b0 ~~~")
    if not np.any(pred_label_arr):
        print("blank")
        pred_b0 = 0
    else:
        pred_b0 = connected_components(pred_label_arr)[0]

    Betti_0_error = abs(pred_b0 - gt_b0)

    return Betti_0_error


def betti_number_error_all_classes(*, gt: sitk.Image, pred: sitk.Image) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with betti_number_error_single_label() as metric_func
    """
    betti_num_err_dict = generate_cls_avg_dict(
        gt=gt,
        pred=pred,
        metric_keys=["B0err"],
        metric_func=betti_number_error_single_label,
    )
    print("\nbetti_number_error_all_classes() =>")
    pprint.pprint(betti_num_err_dict, sort_dicts=False)
    return betti_num_err_dict
