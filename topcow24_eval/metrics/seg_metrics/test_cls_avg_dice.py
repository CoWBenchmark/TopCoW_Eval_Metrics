"""
run the tests with pytest
"""

import math
from pathlib import Path

import pytest
from cls_avg_dice import dice_coefficient_all_classes
from topcow24_eval.constants import BIN_CLASS_LABEL_MAP, MUL_CLASS_LABEL_MAP, TASK
from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

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

TESTDIR_2D = Path("test_assets/seg_metrics/2D")
TESTDIR_3D = Path("test_assets/seg_metrics/3D")


def test_DiceCoefficient_2D_different_dim():
    """
    img shape 6x3 and shape 5x5 should not be able to run
    """
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_zigzag_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    with pytest.raises(AssertionError) as e_info:
        dice_coefficient_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "gt pred not matching shapes!"


def test_DiceCoefficient_2D_binary():
    """
    5x5 2D zigzag shaped binary segmentation comparison
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_zigzag_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_zigzag_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict["1"]["label"] == BIN_CLASS_LABEL_MAP["1"]
    assert math.isclose(dice_dict["1"]["Dice"], (2 * 4) / (9 + 8))
    # ~= 0.47058

    # multi-class segmentation task is also applicable
    # NOTE: but the label should now be BA instead of CoW
    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict["1"]["label"] == MUL_CLASS_LABEL_MAP["1"]
    assert math.isclose(dice_dict["1"]["Dice"], (2 * 4) / (9 + 8))
    # ~= 0.47058


def test_DiceCoefficient_2D_onlyLabel5():
    """
    similar to test_DiceCoefficient_2D_binary
    but now the binary label is label=5 instead of label=1
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_zigzag_label5_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_zigzag_label5_pred.nii.gz"

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
    assert math.isclose(dice_dict["5"]["Dice"], (2 * 13) / (16 + 17))
    # ~= 0.7878

    # the average for this test case should be the same as its own value, since only one class
    assert dice_dict["ClsAvgDice"]["Dice"] == dice_dict["5"]["Dice"]

    # there is an automatic conversion from multi-class to binary
    assert dice_dict["MergedBin"]["Dice"] == dice_dict["5"]["Dice"]


def test_DiceCoefficient_2D_multiclass():
    """
    5x5 2D rings of label 6 and label 4
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_label64_ring_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_label64_ring_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    # check for label 4
    assert dice_dict["4"]["label"] == MUL_CLASS_LABEL_MAP["4"]
    assert math.isclose(dice_dict["4"]["Dice"], (2 * 4) / (8 + 5))
    # ~= 0.6153

    # check for label 6
    assert dice_dict["6"]["label"] == MUL_CLASS_LABEL_MAP["6"]
    assert math.isclose(dice_dict["6"]["Dice"], (2 * 10) / (16 + 11))
    # ~= 0.7407

    # check for average
    assert math.isclose(
        dice_dict["ClsAvgDice"]["Dice"],
        (dice_dict["4"]["Dice"] + dice_dict["6"]["Dice"]) / 2,
    )
    # ~= 0.6780

    # there is an automatic conversion from multi-class to binary
    assert math.isclose(dice_dict["MergedBin"]["Dice"], (2 * 15) / (24 + 16))
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
    gt_path = TESTDIR_2D / "shape_5x5_2D_RGB_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_RBY_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "1": {"label": "BA", "Dice": 1.0},
        "2": {"label": "R-PCA", "Dice": 0.0},
        "3": {"label": "L-PCA", "Dice": 1.0},
        "4": {"label": "R-ICA", "Dice": 0.0},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.5},
        # there is an automatic conversion from multi-class to binary
        "MergedBin": {"label": "MergedBin", "Dice": 2 / 3},
    }


def test_DiceCoefficient_2D_nolabels_binary():
    """
    what if there is no labels in both gt and pred? -> cow=0
    what if there is no labels in gt? -> cow=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict == {"1": {"label": "MergedBin", "Dice": 0}}

    # gt is clean slate, but pred has some predictions
    pred_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    )

    assert dice_dict == {"1": {"label": "MergedBin", "Dice": 0}}


def test_DiceCoefficient_2D_nolabels_multiclass():
    """
    same as test_DiceCoefficient_2D_nolabels_binary but for multiclass

    what if there is no labels in both gt and pred? -> avg=0, cow=0
    what if there is no labels in gt? -> avg=0, cow=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0},
        "MergedBin": {"label": "MergedBin", "Dice": 0},
    }

    # gt is clean slate, but pred has some predictions
    pred_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert dice_dict == {
        "1": {"label": "BA", "Dice": 0},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0},
        # there is an automatic conversion from multi-class to binary
        "MergedBin": {"label": "MergedBin", "Dice": 0},
    }


def test_multi_class_donut():
    """
    a solid donut shape.
        GT has only label-1
        Pred is same shape but one side is label-6
    """
    gt_path = TESTDIR_3D / "shape_5x7x9_3D_1donut.nii.gz"
    pred_path = TESTDIR_3D / "shape_5x7x9_3D_1donut_multiclass.nii.gz"

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
        "1": {"label": "BA", "Dice": 0.7499999999999999},
        "6": {"label": "L-ICA", "Dice": 0.0},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.37499999999999994},
        "MergedBin": {"label": "MergedBin", "Dice": 1.0},
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
    gt_path = TESTDIR_3D / "shape_8x8x8_3D_8Cubes_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_8x8x8_3D_8Cubes_pred.nii.gz"

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
        "1": {"label": "BA", "Dice": 1.0},
        "2": {"label": "R-PCA", "Dice": 0.9333333333333333},
        "3": {"label": "L-PCA", "Dice": 1.0},
        "4": {"label": "R-ICA", "Dice": 0.0},
        "5": {"label": "R-MCA", "Dice": 1.0},
        "6": {"label": "L-ICA", "Dice": 0.8571428571428571},
        "7": {"label": "L-MCA", "Dice": 1.0},
        "8": {"label": "R-Pcom", "Dice": 0.8571428571428571},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.8309523809523809},
        "MergedBin": {"label": "MergedBin", "Dice": 0.8869565217391304},
    }


def test_DiceCoefficient_2D_multiclass_MediumPost():
    """
    using the Dice example from
    https://medium.com/@nghihuynh_37300/understanding-evaluation-metrics-in-medical-image-segmentation-d289a373a3f

    gt label-15, 12, 11
    pred label-15, 11
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_label15_12_11_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_label15_11_pred.nii.gz"

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
    # label 11:
    #        Dice = 2 * 1 / (1 + 3) = 0.5
    # label 12 pred missing:
    #        Dice = 0
    # label 15:
    #        Dice = 2 * 4 / (4 + 5) = 0.888
    # merged binary:
    #        Dice = 0.762 from the MediumPost
    assert dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "11": {"label": "R-ACA", "Dice": 0.5},
        "12": {"label": "L-ACA", "Dice": 0},
        "15": {"label": "3rd-A2", "Dice": 0.888888888888889},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.46296296296296297},
        "MergedBin": {"label": "MergedBin", "Dice": 0.761904761904762},
    }


def test_DiceCoefficient_3D_Fig50():
    """
    example from Fig 50 of
    Common Limitations of Image Processing Metrics: A Picture Story

    images labeled with label-10 and label-15

    label-10 Dice = 0.8
    label-15 Dice = 0.5
    """
    gt_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # multiclass seg task will give the following:
    # label 10:
    #        Dice = .8
    # label 15:
    #        Dice = .5
    # merged binary:
    #        Dice = 2/3
    assert dice_coefficient_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "10": {"label": "Acom", "Dice": 0.8},
        "15": {"label": "3rd-A2", "Dice": 0.5},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.65},
        "MergedBin": {"label": "MergedBin", "Dice": 2 / 3},
    }