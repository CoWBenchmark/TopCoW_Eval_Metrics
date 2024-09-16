import numpy as np
import SimpleITK as sitk

from topcow24_eval.metrics.seg_metrics.clDice import clDice
from topcow24_eval.metrics.seg_metrics.cls_avg_b0 import betti_number_error_all_classes
from topcow24_eval.metrics.seg_metrics.cls_avg_dice import dice_coefficient_all_classes
from topcow24_eval.metrics.seg_metrics.cls_avg_hd95 import hd95_all_classes
from topcow24_eval.metrics.seg_metrics.detection_grp2_labels import (
    detection_grp2_labels,
)
from topcow24_eval.metrics.seg_metrics.generate_cls_avg_dict import update_metrics_dict
from topcow24_eval.metrics.seg_metrics.graph_classification import graph_classification
from topcow24_eval.metrics.seg_metrics.topology_matching import topology_matching


def score_case_task_1_seg(
    *, gt: sitk.Image, pred: sitk.Image, metrics_dict: dict
) -> None:
    """
    score_case() for Task-1-CoW-Segmentation

    work with image gt pred and mutate the metrics_dict object
    """
    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    gt = caster.Execute(gt)
    pred = caster.Execute(pred)

    # make sure they have the same metadata
    # Copies the Origin, Spacing, and Direction from the gt image
    # NOTE: metadata like image.GetPixelIDValue() and image.GetPixelIDTypeAsString()
    # are NOT copied from source image
    pred.CopyInformation(gt)

    # Get arrays. Reordering axis from (z,y,x) to (x,y,z)
    gt_arr = sitk.GetArrayFromImage(gt).transpose((2, 1, 0)).astype(np.uint8)
    pred_arr = sitk.GetArrayFromImage(pred).transpose((2, 1, 0)).astype(np.uint8)

    # Score the case

    # (1) add Dice for each class
    dice_dict = dice_coefficient_all_classes(gt=gt, pred=pred)
    for key in dice_dict:
        update_metrics_dict(
            cls_avg_dict=dice_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="Dice",
        )

    # (2) add clDice
    cl_dice = clDice(v_p_pred=pred_arr, v_l_gt=gt_arr)
    metrics_dict["clDice"] = cl_dice

    # (3) add Betti0 number error for each class
    betti_num_err_dict = betti_number_error_all_classes(gt=gt, pred=pred)
    for key in betti_num_err_dict:
        # no more Betti_1 for topcow24
        # and we are interested in the class-average B0 error per case
        update_metrics_dict(
            cls_avg_dict=betti_num_err_dict,
            metrics_dict=metrics_dict,
            key=key,
            metric_name="B0err",
        )

    # (4) add HD95 and HD for each class
    hd_dict = hd95_all_classes(gt=gt, pred=pred)
    for key in hd_dict:
        # class-average key is singular for HD or HD95
        if key == "ClsAvgHD":
            # HD
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD",
            )
        elif key == "ClsAvgHD95":
            # HD95
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD95",
            )
        else:
            # both HD and HD95 for individual CoW labels
            # HD
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD",
            )
            # HD95
            update_metrics_dict(
                cls_avg_dict=hd_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="HD95",
            )

    # (5) add Group 2 CoW detections
    # each case will have its own detection_dict
    # altogether will be a column of detection_dicts
    # thus name the column as `all_detection_dicts`
    detection_dict = detection_grp2_labels(gt=gt, pred=pred)
    metrics_dict["all_detection_dicts"] = detection_dict

    # (6) add graph classification
    # under the column `all_graph_dicts`
    graph_dict = graph_classification(gt=gt, pred=pred)
    metrics_dict["all_graph_dicts"] = graph_dict

    # (7) add topology matching
    # under the column `all_topo_dicts`
    topo_dict = topology_matching(gt=gt, pred=pred)
    metrics_dict["all_topo_dicts"] = topo_dict
