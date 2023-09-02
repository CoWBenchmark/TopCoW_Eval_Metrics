"""
run the tests with pytest
"""
import math

import pytest

from constants import BIN_CLASS_LABEL_MAP, MUL_CLASS_LABEL_MAP, TASK
from metric_functions import dice_coefficient_all_classes
from utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   _______________________________
# < 1. Tests for Dice and dice_dict >
#   -------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################


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
        dice_coefficient_all_classes(
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

    # there is an automatic conversion from multi-class to binary
    assert math.isclose(dice_dict["CoW"]["dice_score"], (2 * 15) / (24 + 16))
    # = 0.75

    # binary seg task should be invalid for this test!
    # because there are two classes
    with pytest.raises(AssertionError) as e_info:
        dice_coefficient_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "Invalid binary segmentation"


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

    what if there is no labels in both gt and pred? -> avg=0, cow=0
    what if there is no labels in gt? -> avg=0, cow=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "average": {"label": "average", "dice_score": 0},
        "CoW": {"label": "CoW", "dice_score": 0},
    }

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


def test_multi_class_donut():
    """
    a solid donut shape.
        GT has only label-1
        Pred is same shape but one side is label-6
    """
    gt_path = "test_metrics/shape_5x7x9_3D_1donut.nii.gz"
    pred_path = "test_metrics/shape_5x7x9_3D_1donut_multiclass.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # multiclass seg task will give the following:
    # label 1:
    #        Dice = 2 * 18 / (18 + 30) = 0.75
    # label 6:
    #        Dice = 0
    # merged binary:
    #        Dice = 1

    assert dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "dice_score": 0.7499999999999999},
        "6": {"label": "L-ICA", "dice_score": 0.0},
        "average": {"label": "average", "dice_score": 0.37499999999999994},
        "CoW": {"label": "CoW", "dice_score": 1.0},
    }


def test_dice_dict_e2e():
    """
    end-to-end test for dice_dict
    8-label cube made up of 8 of 4x4x4 sub-cubes
        GT: fully filled up
        Pred:
            label 1, 3, 5, 7 fully filled up
            label 2 middle 2x2x2 hollow
            label 4 missing
            label 6, 8 solid donut
    """
    gt_path = "test_metrics/shape_8x8x8_3D_8Cubes_gt.nii.gz"
    pred_path = "test_metrics/shape_8x8x8_3D_8Cubes_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # binary seg task should be invalid for this test!
    # because there are 8 classes!
    with pytest.raises(AssertionError) as e_info:
        dice_coefficient_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "Invalid binary segmentation"

    # multiclass seg task will give the following:
    # label 1, 3, 5, 7 have Dice of 1.0
    # label 2 middle 2x2x2 hollow:
    #        Dice = 2 * 56 / (64 + 56) = 0.93
    # label 4 missing:
    #        Dice = 0
    # label 6, 8 solid donut:
    #        Dice = 2 * 48 / (64 + 48) = 0.857
    # merged binary:
    #        Dice = 2 * (512 -8 -64 -16 -16) / (408 + 512) = 0.8869
    assert dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "dice_score": 1.0},
        "2": {"label": "R-PCA", "dice_score": 0.9333333333333333},
        "3": {"label": "L-PCA", "dice_score": 1.0},
        "4": {"label": "R-ICA", "dice_score": 0.0},
        "5": {"label": "R-MCA", "dice_score": 1.0},
        "6": {"label": "L-ICA", "dice_score": 0.8571428571428571},
        "7": {"label": "L-MCA", "dice_score": 1.0},
        "8": {"label": "R-Pcom", "dice_score": 0.8571428571428571},
        "average": {"label": "average", "dice_score": 0.8309523809523809},
        "CoW": {"label": "CoW", "dice_score": 0.8869565217391304},
    }
