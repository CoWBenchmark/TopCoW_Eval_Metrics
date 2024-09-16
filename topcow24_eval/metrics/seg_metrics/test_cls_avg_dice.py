"""
run the tests with pytest
"""

import math
from pathlib import Path

import pytest
import SimpleITK as sitk
from cls_avg_dice import dice_coefficient_all_classes, dice_coefficient_single_label
from topcow24_eval.constants import MUL_CLASS_LABEL_MAP
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


def test_dice_coefficient_single_label_blank_slices():
    """
    dice_coefficient_single_label() but with blank slices
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    # slice 0 image1 blank, image2 has one label-7
    image2[:, :, 0] = 7
    # slice 1 image1 has two label-8, image2 blank
    image1[2, 0, 1] = 8
    image1[0, 2, 1] = 8
    # slice 2 image1 all label-9, image2 one label-9
    image1[:, :, 2] = 9
    image2[0, 0, 2] = 9

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    label_7_dice = dice_coefficient_single_label(
        gt=image1,
        pred=image2,
        label=7,
    )
    label_8_dice = dice_coefficient_single_label(
        gt=image1,
        pred=image2,
        label=8,
    )
    label_9_dice = dice_coefficient_single_label(
        gt=image1,
        pred=image2,
        label=9,
    )

    assert label_7_dice == 0  # dice = 0 for label-7
    assert label_8_dice == 0  # dice = 0 for label-8
    assert label_9_dice == 0.19999999999999998  # dice = 2/10 for label-9


def test_dice_coefficient_single_label_3D_voxel_spacing():
    """
    with simple 3x3x3 sitk.Image, test that
    dice_coefficient_single_label calculation DONT use spacing information
    """
    label = 42

    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    image1[0, :, :] = label

    image2[:, 2, :] = label

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    # without SetSpacing
    without_spacing = dice_coefficient_single_label(
        gt=image1,
        pred=image2,
        label=label,
    )
    # w/o spacing, DSC = 0.33
    assert round(without_spacing, 5) == round((2 * 3) / (9 + 9), 5)

    # Set Spacing
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # image1 and image2 even have different spacings!
    image1.SetSpacing((1, 3, 5))
    image2.SetSpacing((3, 7, 9))

    image1[0, :, :] = label

    image2[:, 2, :] = label

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    with_spacing = dice_coefficient_single_label(
        gt=image1,
        pred=image2,
        label=label,
    )
    # NOTE: with spacing, even different spacings!
    # DSC is still 0.33
    assert with_spacing == without_spacing


def test_DiceCoefficient_2D_different_dim():
    """
    img shape 6x3 and shape 5x5 should not be able to run
    """
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_zigzag_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    with pytest.raises(AssertionError) as e_info:
        dice_coefficient_all_classes(gt=gt_img, pred=pred_img)
    assert str(e_info.value) == "gt pred not matching shapes!"


def test_DiceCoefficient_2D_binary():
    """
    5x5 2D zigzag shaped binary segmentation comparison
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_zigzag_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_zigzag_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # multi-class segmentation task is also applicable
    # NOTE: but the label should now be BA instead of CoW
    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=pred_img)

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

    # this test should only work for multiclass task even though it only has one label
    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=pred_img)

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

    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=pred_img)

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


def test_DiceCoefficient_2D_nonOverlapped_multiclass():
    """
    5x5 2D multiclass gt is three columns of label(1,2,3)
    pred is three columns of label(1,3,4)
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_RGB_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_RBY_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=pred_img)

    assert dice_dict == {
        "1": {"label": "BA", "Dice": 1.0},
        "2": {"label": "R-PCA", "Dice": 0.0},
        "3": {"label": "L-PCA", "Dice": 1.0},
        "4": {"label": "R-ICA", "Dice": 0.0},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.5},
        # there is an automatic conversion from multi-class to binary
        "MergedBin": {"label": "MergedBin", "Dice": 2 / 3},
    }


def test_DiceCoefficient_2D_nolabels_task_multiclass():
    """
    what if there is no labels in both gt and pred? -> avg=0, cow=0
    what if there is no labels in gt? -> avg=0, cow=0
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=gt_img)

    assert dice_dict == {
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0},
        # there is an automatic conversion from multi-class to binary
        "MergedBin": {"label": "MergedBin", "Dice": 0},
    }

    # gt is clean slate, but pred has some predictions
    pred_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    dice_dict = dice_coefficient_all_classes(gt=gt_img, pred=pred_img)

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

    assert dice_coefficient_all_classes(gt=gt_img, pred=pred_img) == {
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
    assert dice_coefficient_all_classes(gt=gt_img, pred=pred_img) == {
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

    # multiclass seg task will give the following:
    # label 11:
    #        Dice = 2 * 1 / (1 + 3) = 0.5
    # label 12 pred missing:
    #        Dice = 0
    # label 15:
    #        Dice = 2 * 4 / (4 + 5) = 0.888
    # merged binary:
    #        Dice = 0.762 from the MediumPost
    assert dice_coefficient_all_classes(gt=gt_img, pred=pred_img) == {
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
    assert dice_coefficient_all_classes(gt=gt_img, pred=pred_img) == {
        "10": {"label": "Acom", "Dice": 0.8},
        "15": {"label": "3rd-A2", "Dice": 0.5},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.65},
        "MergedBin": {"label": "MergedBin", "Dice": 2 / 3},
    }


def test_DiceCoefficient_topcow023mr():
    """
    topcow mr 023 vs LPS_ICA_PCA_flipped.nii.gz

    Despite the slight difference in their directions,
    the OverlapMeasures should be run

    topcow_mr_roi_023.nii.gz
    image.GetDirection() = (0.9998825394863241, -4.957000000637633e-12,
    -0.015326684246574056, -2.8804510733315957e-06, 0.9999999823397805,
    -0.00018791525753699667, 0.01532668333601454, 0.00018793732625326786,
    0.9998825218183695)

    LPS_ICA_PCA_flipped.nii.gz
    image.GetDirection() = (0.9998825394653973, -1.440228107624726e-06,
    -0.015326684246542236, -1.440228121515313e-06, 0.9999999823408161,
    -0.00018792630485902698, 0.015326684904237034, 0.0001879262974336947,
    0.9998825218162936)
    """
    testdir = Path("test_assets/seg_metrics/topcow_roi")

    gt, _ = load_image_and_array_as_uint8(testdir / "topcow_mr_roi_023.nii.gz", True)
    pred, _ = load_image_and_array_as_uint8(
        testdir / "LPS_ICA_PCA_flipped.nii.gz", True
    )

    # has a tiny blob of label-6 overlap due to the L-ICA outlier
    assert dice_coefficient_all_classes(gt=gt, pred=pred) == {
        "1": {"label": "BA", "Dice": 0.0},
        "2": {"label": "R-PCA", "Dice": 0.0},
        "3": {"label": "L-PCA", "Dice": 0.0},
        "4": {"label": "R-ICA", "Dice": 0.0},
        "5": {"label": "R-MCA", "Dice": 0},
        "6": {"label": "L-ICA", "Dice": 0.005530520278574354},
        "7": {"label": "L-MCA", "Dice": 0},
        "8": {"label": "R-Pcom", "Dice": 0},
        "10": {"label": "Acom", "Dice": 0},
        "11": {"label": "R-ACA", "Dice": 0},
        "12": {"label": "L-ACA", "Dice": 0},
        "ClsAvgDice": {"label": "ClsAvgDice", "Dice": 0.0005027745707794867},
        "MergedBin": {"label": "MergedBin", "Dice": 0.025192476366825846},
    }
