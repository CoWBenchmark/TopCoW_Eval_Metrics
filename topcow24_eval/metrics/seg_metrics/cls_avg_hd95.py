"""
Class-average Hausdorff Distance 95% Percentile (HD95)

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from enum import Enum

import numpy as np
import SimpleITK as sitk
from generate_cls_avg_dict import generate_cls_avg_dict
from SimpleITK import GetArrayViewFromImage as ArrayView
from topcow24_eval.constants import HD95_UPPER_BOUND
from topcow24_eval.utils.utils_mask import arr_is_binary


def hd95_single_label(*, gt: sitk.Image, pred: sitk.Image, label: int) -> list[float]:
    """
    Calculates the Hausdorff distance at 95% percentile

    NOTE: While there are many different implementations,
    packages, and even definitions(!) to calculate HD95,
    we decide to go with the definiton from
        Reinke, A., Tizabi, M.D., Baumgartner, M. et al.
        Understanding metric-related pitfalls in image analysis validation.
        Nat Methods 21, 182–194 (2024).
        See Fig. SN 3.63 and
        https://metrics-reloaded.dkfz.de/metric?id=hd95
    The implementation takes the max of two d_95:
        max(d_95(A,B), d_95(B,A))
    We verified this implementation with various Figs from
        Reinke et al., 2021
        Common Limitations of Image Processing Metrics: A Picture Story

    NOTE: in case of missing values (FP or FN), set the HD95
    to be roughly the maximum distance in ROI = 90 mm (HD95_UPPER_BOUND)

    Parameters
    ----------
    gt:
        ground truth mask sitk image
    pred:
        predicted mask sitk image
    label:
        annotation label integer

    Returns
    ----------
    [float hd95_score, float hd100_score]
    The distance unit is the same as the voxelspacing,
        which is usually in mm.

    References:
        Reinke et al., 2024
            Metrics reloaded: recommendations for image analysis validation
        Reinke et al., 2021
            Common Limitations of Image Processing Metrics: A Picture Story
        ITK forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832
        ITK tutorial surface_hausdorff_distance:
            http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py
    """
    print(f"\n--> hd95_single_label(label-{label})\n")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    # only need bool binary mask of the current label
    gt_label_img = gt == label
    pred_label_img = pred == label

    # gt_arr and pred_arr are from union of showed-up labels,
    # thus they will not be both all zeros
    # thus only FP and FN can happen
    # handle FP and FN with HD95_UPPER_BOUND

    gt_label_arr = sitk.GetArrayFromImage(gt_label_img)
    pred_label_arr = sitk.GetArrayFromImage(pred_label_img)

    # make sure the masks are binary
    assert arr_is_binary(gt_label_arr), "hd95_single_label expects binary gt_arr"
    assert arr_is_binary(pred_label_arr), "hd95_single_label expects binary pred_arr"

    # check if either gt or pred label_arr is all zero
    if (not np.any(gt_label_arr)) or (not np.any(pred_label_arr)):
        print(f"[!!Warning] label-{label} empty for gt or pred")
        return [HD95_UPPER_BOUND, HD95_UPPER_BOUND]

    ##################################################################
    # Now the real HD95 implementation :)
    # -> max(d_95(A,B), d_95(B,A))
    ##################################################################

    # get the distance_map, surface, and number of surface pixels
    # for both gt/ref and pred
    (
        ref_distance_map,
        ref_surface,
        num_ref_surface_pixels,
    ) = _get_surface_distance(gt_label_img)
    (
        pred_distance_map,
        pred_surface,
        num_pred_surface_pixels,
    ) = _get_surface_distance(pred_label_img)

    # extract the distances of boundary_ref to boundary_pred
    # and vice versa for both directions
    # NOTE: SimpleITK MultiplyImageFilter requires
    # both input images to have the same pixel type
    # distance_map is float32, so need to cast surface to float
    ref2pred_distance_map = pred_distance_map * sitk.Cast(ref_surface, sitk.sitkFloat32)
    pred2ref_distance_map = ref_distance_map * sitk.Cast(pred_surface, sitk.sitkFloat32)

    # with np.printoptions(precision=1, suppress=True):
    #     print("ref2pred_distance_map =\n", ArrayView(ref2pred_distance_map))
    #     print("pred2ref_distance_map =\n", ArrayView(pred2ref_distance_map))

    # extract the non-zero distances from the distance_map
    ref2pred_distances = list(
        ArrayView(ref2pred_distance_map)[ArrayView(ref2pred_distance_map) != 0]
    )
    # create a list based on the number of surface pixels
    # populate the rest of the list with 0
    ref2pred_distances += [0] * (num_ref_surface_pixels - len(ref2pred_distances))

    # print("ref2pred_distances =\n", ref2pred_distances)
    # print("# ref2pred_distances =\n", len(ref2pred_distances))

    # do the same for ther other direction pred2ref
    pred2ref_distances = list(
        ArrayView(pred2ref_distance_map)[ArrayView(pred2ref_distance_map) != 0]
    )
    pred2ref_distances += [0] * (num_pred_surface_pixels - len(pred2ref_distances))

    # print("pred2ref_distances =\n", pred2ref_distances)
    # print("# pred2ref_distances =\n", len(pred2ref_distances))

    # use formula -> max(d_95(A,B), d_95(B,A))
    d_95_ref2pred = np.percentile(ref2pred_distances, 95)
    # print("d_95_ref2pred = ", d_95_ref2pred)
    d_95_pred2ref = np.percentile(pred2ref_distances, 95)
    # print("d_95_pred2ref = ", d_95_pred2ref)

    hd95_score = max(d_95_ref2pred, d_95_pred2ref)
    print("hd95_score = ", hd95_score)

    # also keep track of HD max
    d_100_ref2pred = np.percentile(ref2pred_distances, 100)
    # print("d_100_ref2pred = ", d_100_ref2pred)
    d_100_pred2ref = np.percentile(pred2ref_distances, 100)
    # print("d_100_pred2ref = ", d_100_pred2ref)
    hd100_score = max(d_100_ref2pred, d_100_pred2ref)
    print("hd100_score = ", hd100_score)

    return [hd95_score, hd100_score]


def _get_surface_distance(seg: sitk.Image) -> tuple[sitk.Image, sitk.Image, int]:
    """
    Code adapted from:
        ITK Forum:
            https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/
        ITK tutorial surface_hausdorff_distance:
            http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
        seg-metrics: a Python package to compute segmentation metrics
            https://github.com/Jingnan-Jia/segmentation_metrics
        ToothFairy1 Challenge:
            https://github.com/AImageLab-zip/ToothFairy/blob/main/ToothFairy/evaluation/evaluation.py
    """

    # get map of the distance to boundary for input segmentation mask
    # use image spacing with Maurer distance transform
    seg_distance_map = sitk.Abs(
        sitk.SignedMaurerDistanceMap(
            seg,
            squaredDistance=False,
            useImageSpacing=True,
        )
    )

    # extract the contour outline for later masking
    seg_surface = sitk.LabelContour(
        seg,
        # set to fully connected
        fullyConnected=True,
    )

    # with np.printoptions(precision=1, suppress=True):
    #     print("seg_distance_map =\n", ArrayView(seg_distance_map))
    #     print("seg_surface =\n", ArrayView(seg_surface))

    # get the number of surface pixels for HD sorting later
    statistics_image_filter = sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(seg_surface)

    num_surface_pixels = int(statistics_image_filter.GetSum())
    # print("num_surface_pixels = ", num_surface_pixels)

    return seg_distance_map, seg_surface, num_surface_pixels


def hd95_all_classes(*, gt: sitk.Image, pred: sitk.Image, task: Enum) -> dict:
    """
    use the dict generator from generate_cls_avg_dict
    with hd95_single_label() as metric_func
    """
    hd_dict = generate_cls_avg_dict(
        gt=gt,
        pred=pred,
        task=task,
        # hd95_single_label returns [hd95_score, hd100_score]
        metric_keys=["HD95", "HD"],
        metric_func=hd95_single_label,
    )
    print("\nhd95_all_classes() =>")
    pprint.pprint(hd_dict, sort_dicts=False)
    return hd_dict
