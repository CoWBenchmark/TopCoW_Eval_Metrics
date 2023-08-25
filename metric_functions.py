from enum import Enum
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from skimage.morphology import skeletonize, skeletonize_3d

from constants import BIN_CLASS_LABEL_MAP, MUL_CLASS_LABEL_MAP, TASK


def convert_multiclass_to_binary(array: np.array) -> np.array:
    """merge all non-background labels into binary class for clDice"""
    array[array > 1] = 1
    return array.astype(bool)


def extract_labels(*, gt_array: np.array, pred_array: np.array) -> List:
    """Extracts union of labels in gt and pred masks"""
    labels_gt = np.unique(gt_array)
    labels_pred = np.unique(pred_array)
    labels = list(set().union(labels_gt, labels_pred))
    labels = [int(x) for x in labels]
    return labels


def dice_coefficient_single_label(
    *, gt: sitk.Image, pred: sitk.Image, label: int
) -> float:
    """use overlap measures filter with a single label"""
    overlap_measures = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures.SetNumberOfThreads(1)
    overlap_measures.Execute(gt, pred)
    dice_score = overlap_measures.GetDiceCoefficient(label)
    return dice_score


def dice_coefficient_all_classes(
    *, gt: sitk.Image, pred: sitk.Image, task: Enum
) -> Dict:
    """
    If task is TASK.BINARY_SEGMENTATION, it will compute the CoW class dice score
    If task is TASK.MULTICLASS_SEGMENTATION, it will
        compute dice scores of union of classes and an overall average per case.

    NOTE: returned dice_dict only considers all labels
        which are present in both gt and pred to compute the per-case-average
    """
    # NOTE: SimpleITK npy axis ordering is (z,y,x)!
    # reorder from (z,y,x) to (x,y,z)
    gt_array = sitk.GetArrayFromImage(gt).transpose((2, 1, 0)).astype(np.uint8)
    pred_array = sitk.GetArrayFromImage(pred).transpose((2, 1, 0)).astype(np.uint8)

    # SimpleITK image GetSize should have the same shape as transposed (x,y,z) npy
    assert (
        gt.GetSize() == gt_array.shape
    ), f"gt.GetSize():{gt.GetSize()} != gt_array.shape:{gt_array.shape}"
    assert (
        pred.GetSize() == pred_array.shape
    ), f"pred.GetSize():{pred.GetSize()} != pred_array.shape:{pred_array.shape}"

    labels = extract_labels(gt_array=gt_array, pred_array=pred_array)
    labels.remove(0)
    print(f"labels = {labels}")

    dice_dict = {}

    if task == TASK.BINARY_SEGMENTATION:
        # when there are no labels in the ROI, return blank dice_dict
        if len(labels) == 0:
            return {"1": {"label": BIN_CLASS_LABEL_MAP["1"], "dice_score": 0}}
        # otherwise compute the dice for CoW and update the dice_dict

        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentation"
        dice_score = dice_coefficient_single_label(gt=gt, pred=pred, label=1)
        dice_dict["1"] = {"label": BIN_CLASS_LABEL_MAP["1"], "dice_score": dice_score}
    else:
        # when there are no labels in the ROI,
        # return blank dice_dict with only average of 0
        if len(labels) == 0:
            return {
                "average": {"label": "average", "dice_score": 0},
            }
        # otherwise compute the dice for label in union and update the dice_dict

        sum_scores = 0

        for label in labels:
            # update the dice_dict for that label
            dice_score = dice_coefficient_single_label(gt=gt, pred=pred, label=label)
            dice_dict[str(label)] = {
                "label": MUL_CLASS_LABEL_MAP[str(label)],
                "dice_score": dice_score,
            }
            sum_scores += dice_score

        overall_dice = sum_scores / len(labels)
        dice_dict["average"] = {"label": "average", "dice_score": overall_dice}

        # multi-class segmentation is also automatically considered for binary task
        # binary task score is done by binary-thresholding the sitk Image
        gt_bin = sitk.BinaryThreshold(
            gt,
            lowerThreshold=1,
        )
        pred_bin = sitk.BinaryThreshold(
            pred,
            lowerThreshold=1,
        )
        dice_score = dice_coefficient_single_label(gt=gt_bin, pred=pred_bin, label=1)
        dice_dict[BIN_CLASS_LABEL_MAP["1"]] = {
            "label": BIN_CLASS_LABEL_MAP["1"],
            "dice_score": dice_score,
        }

    print(f"dice_coefficient_all_classes dice_dict = {dice_dict}")
    return dice_dict


def cl_score(*, s_skeleton, v_image):
    """[this function computes the skeleton volume overlap]
    Args:
        s ([bool]): [skeleton]
        v ([bool]): [image]
    Returns:
        [float]: [computed skeleton volume intersection]

    meanings of v, s refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """
    if np.sum(s_skeleton) == 0:
        return 0
    return np.sum(s_skeleton * v_image) / np.sum(s_skeleton)


def clDice(*, v_p_pred, v_l_gt):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]

    meanings of v_l, v_p, s_l, s_p refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0

    return 2 * tprec * tsens / (tprec + tsens)
