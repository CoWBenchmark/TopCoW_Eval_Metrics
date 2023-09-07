from enum import Enum
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
from skimage.measure import euler_number, label
from skimage.morphology import skeletonize, skeletonize_3d

from constants import BIN_CLASS_LABEL_MAP, MUL_CLASS_LABEL_MAP, TASK


def convert_multiclass_to_binary(array: np.array) -> np.array:
    """merge all non-background labels into binary class for clDice"""
    bin_merged = np.where(array > 0, True, False)
    return bin_merged.astype(bool)


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
    If task is TASK.BINARY_SEGMENTATION,
        it will compute the CoW class dice score
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
        # return blank dice_dict with only average and merged_binary of 0
        if len(labels) == 0:
            return {
                "average": {"label": "average", "dice_score": 0},
                BIN_CLASS_LABEL_MAP["1"]: {
                    "label": BIN_CLASS_LABEL_MAP["1"],
                    "dice_score": 0,
                },
            }
        # otherwise compute the dice for label in union and update the dice_dict

        sum_scores = 0

        for voxel_label in labels:
            # update the dice_dict for that label
            dice_score = dice_coefficient_single_label(
                gt=gt, pred=pred, label=voxel_label
            )
            dice_dict[str(voxel_label)] = {
                "label": MUL_CLASS_LABEL_MAP[str(voxel_label)],
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


def filter_mask_by_label(mask: np.array, label: int) -> np.array:
    """
    filter the mask (numpy array), keep the voxels matching the label as 1
        convert the voxels that are not matching the label as 0
    """
    return np.where(mask == label, 1, 0)


def betti_number(img: np.array) -> List:
    """
    calculates the Betti number B0, B1, and B2 for a 3D img
    from the Euler characteristic number

    code prototyped by
    - Martin Menten (Imperial College)
    - Suprosanna Shit (Technical University Munich)
    - Johannes C. Paetzold (Imperial College)
    """

    # make sure the image is 3D (for connectivity settings)
    assert img.ndim == 3

    # 6 or 26 neighborhoods are defined for 3D images,
    # (connectivity 1 and 3, respectively)
    # If foreground is 26-connected, then background is 6-connected, and conversely
    N6 = 1
    N26 = 3

    # important first step is to
    # pad the image with background (0) around the border!
    padded = np.pad(img, pad_width=1)

    # make sure the image is binary with
    assert set(np.unique(padded)).issubset({0, 1})

    # calculate the Betti numbers B0, B2
    # then use Euler characteristic to get B1

    # get the label connected regions for foreground
    _, b0 = label(
        padded,
        # return the number of assigned labels
        return_num=True,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    euler_char_num = euler_number(
        padded,
        # 26 neighborhoods for foreground
        connectivity=N26,
    )

    # get the label connected regions for background
    _, b2 = label(
        1 - padded,
        # return the number of assigned labels
        return_num=True,
        # 6 neighborhoods for background
        connectivity=N6,
    )

    # NOTE: need to substract 1 from b2
    b2 -= 1

    b1 = b0 + b2 - euler_char_num  # Euler number = Betti:0 - Bett:1 + Betti:2

    print(f"Betti number: b0 = {b0}, b1 = {b1}, b2 = {b2}")

    return [b0, b1, b2]


def betti_number_error_all_classes(
    *, gt: sitk.Image, pred: sitk.Image, task: Enum
) -> Dict:
    """
    If task is TASK.BINARY_SEGMENTATION,
        it will compute the CoW class betti number error
    If task is TASK.MULTICLASS_SEGMENTATION, it will
        compute betti number errors of union of classes and an overall average per case.

    NOTE: returned betti_num_err_dict only considers all labels
        which are present in both gt and pred to compute the per-case-average
    """
    # print("\n-- call betti_number_error_all_classes()")

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

    betti_num_err_dict = {}

    if task == TASK.BINARY_SEGMENTATION:
        # when there are no labels in the ROI, return blank betti_num_err_dict
        if len(labels) == 0:
            return {
                "1": {
                    "label": BIN_CLASS_LABEL_MAP["1"],
                    "Betti_0_error": 0,
                    "Betti_1_error": 0,
                }
            }
        # otherwise compute the Betti number error and update the betti_num_err_dict

        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentation"

        print("\nvoxel_label = CoW_binary")
        print("~~~ gt_betti_numbers ~~~")
        gt_betti_numbers = betti_number(gt_array)
        print("~~~ pred_betti_numbers ~~~")
        pred_betti_numbers = betti_number(pred_array)

        Betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        Betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        betti_num_err_dict["1"] = {
            "label": BIN_CLASS_LABEL_MAP["1"],
            "Betti_0_error": Betti_0_error,
            "Betti_1_error": Betti_1_error,
        }
    else:
        # when there are no labels in the ROI,
        # return blank betti_num_err_dict with only B0err_average and merged_binary of 0
        if len(labels) == 0:
            return {
                "B0err_average": {"label": "B0err_average", "Betti_0_error": 0},
                BIN_CLASS_LABEL_MAP["1"]: {
                    "label": BIN_CLASS_LABEL_MAP["1"],
                    "Betti_0_error": 0,
                    "Betti_1_error": 0,
                },
            }
        # otherwise compute the Betti 0 number error for label in union
        # and update the betti_num_err_dict

        sum_scores = 0

        for voxel_label in labels:
            print("\nvoxel_label = ", voxel_label)
            # filter the view by that label
            filtered_gt = filter_mask_by_label(gt_array, voxel_label)
            filtered_pred = filter_mask_by_label(pred_array, voxel_label)

            print("~~~ gt_betti_numbers ~~~")
            gt_betti_numbers = betti_number(filtered_gt)
            print("~~~ pred_betti_numbers ~~~")
            pred_betti_numbers = betti_number(filtered_pred)

            Betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])

            # update the betti_num_err_dict for that label
            betti_num_err_dict[str(voxel_label)] = {
                "label": MUL_CLASS_LABEL_MAP[str(voxel_label)],
                "Betti_0_error": Betti_0_error,
            }
            sum_scores += Betti_0_error

        overall_Betti_0_error = sum_scores / len(labels)
        betti_num_err_dict["B0err_average"] = {
            "label": "B0err_average",
            "Betti_0_error": overall_Betti_0_error,
        }

        print("\nvoxel_label = merged_binary")
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

        # NOTE: SimpleITK npy axis ordering is (z,y,x)!
        # reorder from (z,y,x) to (x,y,z)
        gt_bin_array = (
            sitk.GetArrayFromImage(gt_bin).transpose((2, 1, 0)).astype(np.uint8)
        )
        pred_bin_array = (
            sitk.GetArrayFromImage(pred_bin).transpose((2, 1, 0)).astype(np.uint8)
        )

        print("~~~ gt_betti_numbers ~~~")
        gt_betti_numbers = betti_number(gt_bin_array)
        print("~~~ pred_betti_numbers ~~~")
        pred_betti_numbers = betti_number(pred_bin_array)

        Betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        Betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])

        betti_num_err_dict[BIN_CLASS_LABEL_MAP["1"]] = {
            "label": BIN_CLASS_LABEL_MAP["1"],
            "Betti_0_error": Betti_0_error,
            "Betti_1_error": Betti_1_error,
        }

    print(f"betti_number_error_all_classes betti_num_err_dict = {betti_num_err_dict}")
    return betti_num_err_dict
