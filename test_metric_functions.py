"""
run the tests with pytest
"""
import math

import pytest
from skimage.morphology import skeletonize

from constants import BIN_CLASS_LABEL_MAP, MUL_CLASS_LABEL_MAP, TASK
from metric_functions import (
    cl_score,
    clDice,
    convert_multiclass_to_binary,
    dice_coefficient_all_classes,
)
from utils_nii_mha_sitk import load_image_and_array_as_uint8


def test_DiceCoefficient_2D_binary():
    """
    5x5 2D zigzag shaped binary segmentation comparison
    """
    gt_path = "test_metrics/shape_5x5_2D_zigzag_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_zigzag_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict["1"]["label"] == BIN_CLASS_LABEL_MAP["1"]
    assert math.isclose(dice_dict["1"]["dice_score"], (2 * 4) / (9 + 8))
    # ~= 0.47058

    # multi-class segmentation task is also applicable
    # NOTE: but the label should now be BA instead of CoW
    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict["1"]["label"] == MUL_CLASS_LABEL_MAP["1"]
    assert math.isclose(dice_dict["1"]["dice_score"], (2 * 4) / (9 + 8))
    # ~= 0.47058


def test_DiceCoefficient_2D_onlyLabel5():
    """
    similar to test_DiceCoefficient_2D_binary
    but now the binary label is label=5 instead of label=1
    """
    gt_path = "test_metrics/shape_5x5_2D_zigzag_label5_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_zigzag_label5_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # binary seg task should be invalid for this test!
    # even though it only has one label class of 5
    with pytest.raises(AssertionError) as e_info:
        dice_dict = dice_coefficient_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "Invalid binary segmentation"

    # this test should only work for multiclass task even though it only has one label
    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict["5"]["label"] == MUL_CLASS_LABEL_MAP["5"]
    assert math.isclose(dice_dict["5"]["dice_score"], (2 * 13) / (16 + 17))
    # ~= 0.7878

    # the average for this test case should be the same as its own value, since only one class
    assert dice_dict["average"]["dice_score"] == dice_dict["5"]["dice_score"]

    # there is an automatic conversion from multi-class to binary
    assert dice_dict["CoW"]["dice_score"] == dice_dict["5"]["dice_score"]


def test_DiceCoefficient_2D_multiclass():
    """
    5x5 2D rings of label 6 and label 4
    """
    gt_path = "test_metrics/shape_5x5_2D_label64_ring_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_label64_ring_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    # check for label 4
    assert dice_dict["4"]["label"] == MUL_CLASS_LABEL_MAP["4"]
    assert math.isclose(dice_dict["4"]["dice_score"], (2 * 4) / (8 + 5))
    # ~= 0.6153

    # check for label 6
    assert dice_dict["6"]["label"] == MUL_CLASS_LABEL_MAP["6"]
    assert math.isclose(dice_dict["6"]["dice_score"], (2 * 10) / (16 + 11))
    # ~= 0.7407

    # check for average
    assert math.isclose(
        dice_dict["average"]["dice_score"],
        (dice_dict["4"]["dice_score"] + dice_dict["6"]["dice_score"]) / 2,
    )
    # ~= 0.6780

    # binary seg task should be invalid for this test!
    # because there are two classes
    with pytest.raises(AssertionError) as e_info:
        dice_dict = dice_coefficient_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "Invalid binary segmentation"

    # there is an automatic conversion from multi-class to binary
    assert math.isclose(dice_dict["CoW"]["dice_score"], (2 * 15) / (24 + 16))
    # = 0.75


def test_DiceCoefficient_2D_nonOverlapped_multiclass():
    """
    5x5 2D multiclass gt is three columns of label(1,2,3)
    pred is three columns of label(1,3,4)
    """
    gt_path = "test_metrics/shape_5x5_2D_RGB_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_RBY_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "1": {"label": "BA", "dice_score": 1.0},
        "2": {"label": "R-PCA", "dice_score": 0.0},
        "3": {"label": "L-PCA", "dice_score": 1.0},
        "4": {"label": "R-ICA", "dice_score": 0.0},
        "average": {"label": "average", "dice_score": 0.5},
        # there is an automatic conversion from multi-class to binary
        "CoW": {"label": "CoW", "dice_score": 2 / 3},
    }


def test_DiceCoefficient_2D_nolabels_binary():
    """
    what if there is no labels in both gt and pred? -> cow=0
    what if there is no labels in gt? -> cow=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict == {"1": {"label": "CoW", "dice_score": 0}}

    # gt is clean slate, but pred has some predictions
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict == {"1": {"label": "CoW", "dice_score": 0}}


def test_DiceCoefficient_2D_nolabels_multiclass():
    """
    same as test_DiceCoefficient_2D_nolabels_binary but for multiclass

    what if there is no labels in both gt and pred? -> avg=0
    what if there is no labels in gt? -> avg=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {"average": {"label": "average", "dice_score": 0}}

    # gt is clean slate, but pred has some predictions
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "1": {"label": "BA", "dice_score": 0},
        "average": {"label": "average", "dice_score": 0},
        # there is an automatic conversion from multi-class to binary
        "CoW": {"label": "CoW", "dice_score": 0},
    }


def test_cl_score_2D_blob():
    """
    6x3 2D with an elongated blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = "test_metrics/shape_6x3_2D_clDice_elong_gt.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    _, gt_arr = load_image_and_array_as_uint8(gt_path)
    _, pred_arr = load_image_and_array_as_uint8(pred_path)

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(pred_arr)
    gt_mask = convert_multiclass_to_binary(gt_arr)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (6 / 6)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (4 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_arr, v_l_gt=gt_arr) == 1


def test_cl_score_2D_Tshaped():
    """
    5x5 2D with a T-shaped blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = "test_metrics/shape_5x5_2D_clDice_Tshaped_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_clDice_Tshaped_pred.nii.gz"

    _, gt_arr = load_image_and_array_as_uint8(gt_path)
    _, pred_arr = load_image_and_array_as_uint8(pred_path)

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(pred_arr)
    gt_mask = convert_multiclass_to_binary(gt_arr)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (5 / 5)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (3 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_arr, v_l_gt=gt_arr) == (3 / 2) / (7 / 4)
    # ~= 0.85714

    """
    same as test_cl_score_2D_Tshaped but on multiclass
    """
    # with multiclass labels
    multiclass_gt_path = "test_metrics/shape_5x5_2D_clDice_Tshaped_multiclass_gt.nii.gz"
    multiclass_pred_path = (
        "test_metrics/shape_5x5_2D_clDice_Tshaped_multiclass_pred.nii.gz"
    )
    _, multiclass_gt_arr = load_image_and_array_as_uint8(multiclass_gt_path)
    _, multiclass_pred_arr = load_image_and_array_as_uint8(multiclass_pred_path)
    multiclass_pred_mask = convert_multiclass_to_binary(multiclass_pred_arr)
    multiclass_gt_mask = convert_multiclass_to_binary(multiclass_gt_arr)

    # test_cl_score_2D_Tshaped should match test_cl_score_2D_Tshaped with multiclass!
    assert tprec == cl_score(
        s_skeleton=skeletonize(multiclass_pred_mask), v_image=multiclass_gt_mask
    )

    assert tsens == cl_score(
        s_skeleton=skeletonize(multiclass_gt_mask), v_image=multiclass_pred_mask
    )

    assert clDice(v_p_pred=pred_arr, v_l_gt=gt_arr) == clDice(
        v_p_pred=multiclass_pred_arr, v_l_gt=multiclass_gt_arr
    )
    assert clDice(v_p_pred=multiclass_pred_arr, v_l_gt=multiclass_gt_arr) == (
        (3 / 2) / (7 / 4)
    )
    # ~= 0.85714


# def test_BettiNumber
