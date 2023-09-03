"""
run the tests with pytest
"""
import numpy as np
import pytest

from constants import TASK
from metric_functions import betti_number, betti_number_error_all_classes
from utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   ________________________________
# < 3. Tests for Betti Number Errors >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################


def test_no_betti():
    """
    all zeros should give Betti numbers: 0 0 0
    """
    assert betti_number(np.zeros((42, 42, 42))) == [0, 0, 0]


def test_one_betti():
    """
    all ones should give Betti numbers: 1 0 0
    """
    assert betti_number(np.ones((42, 42, 42))) == [1, 0, 0]


def test_singleBlob_noBorder():
    """
    GT is a 2x2x2 blob all filled with label 1 and no border (8 cubes)
    Pred is the same blob but missing a corner voxel (7 cubes)
    """
    gt_path = "test_metrics/betti_num_singleBlob_2x2x2_gt.nii.gz"
    pred_path = "test_metrics/betti_num_singleBlob_2x2x2_pred.nii.gz"

    _, gt_mask = load_image_and_array_as_uint8(gt_path)
    _, pred_mask = load_image_and_array_as_uint8(pred_path)

    gt_betti_numbers = betti_number(gt_mask)
    assert gt_betti_numbers == [1, 0, 0]

    pred_betti_numbers = betti_number(pred_mask)
    assert pred_betti_numbers == [1, 0, 0]


def test_singleBlob_noBorder_betti_num_err_dict():
    """
    end to end test for betti_num_err_dict based on test_singleBlob_noBorder()
    """
    gt_path = "test_metrics/betti_num_singleBlob_2x2x2_gt.nii.gz"
    pred_path = "test_metrics/betti_num_singleBlob_2x2x2_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # label-1 for binary segmentation task is CoW
    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    ) == {"1": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0}}

    # label-1 for multiclass segmentation task is BA
    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "Betti_0_error": 0},
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 0.0},
        "CoW": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0},
    }


def test_twoIslands():
    """
    two islands of label 1
    Answer: B0=2 B1=0 B2=0
    """
    # touching the edge, i.e. no border
    mask_path = "test_metrics/shape_3x4x2_3D_twoIslands.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    betti_numbers = betti_number(mask)

    assert betti_numbers == [2, 0, 0]

    # padded manually (NOTE betti_number() pads too)
    mask_path = "test_metrics/shape_5x6x4_3D_twoIslands_padded.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    betti_numbers = betti_number(mask)

    assert betti_numbers == [2, 0, 0]


def test_multi_class_twoIslands():
    """
    similar to test_twoIslands() but have pred has two labels
    """
    gt_path = "test_metrics/shape_3x4x2_3D_twoIslands.nii.gz"
    pred_path = "test_metrics/shape_3x4x2_3D_twoIslands_multiclass.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # GT has B0=2, B1=0, B2=0 and only has label-1
    #
    # Pred has the same Betti numbers for:
    # label 1 and label 2 and merged binary

    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "Betti_0_error": 0},
        "2": {"label": "R-PCA", "Betti_0_error": 2},
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 1},
        "CoW": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0},
    }


def test_donut():
    """
    a solid donut
    Answer: B0 = 1, B1 = 1, B2 = 0
    """
    mask_path = "test_metrics/shape_5x7x9_3D_1donut.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    betti_numbers = betti_number(mask)

    assert betti_numbers == [1, 1, 0]


def test_multi_class_donut():
    """
    similar to test_donut() but have pred has two labels
    """
    gt_path = "test_metrics/shape_5x7x9_3D_1donut.nii.gz"
    pred_path = "test_metrics/shape_5x7x9_3D_1donut_multiclass.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # GT has B0=1, B1=1, B2=0 and only has label-1
    #
    # Pred has the same Betti numbers for merged binary
    # but label-1 and label-6 should have B0=1, B1=0, B2=0

    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "Betti_0_error": 0},
        "6": {"label": "L-ICA", "Betti_0_error": 1},
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 1 / 2},
        "CoW": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0},
    }


def test_cross():
    """
    cross with a hole in the middle, sort of like a solid donut
    Answer same as solid donut
    """
    # touching the edge, i.e. no border
    mask_path = "test_metrics/shape_3x4x2_3D_cross.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    betti_numbers = betti_number(mask)

    assert betti_numbers == [1, 1, 0]

    # padded manually (NOTE betti_number() pads too)
    mask_path = "test_metrics/shape_5x6x4_3D_cross_padded.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    betti_numbers = betti_number(mask)

    assert betti_numbers == [1, 1, 0]


def test_betti_num_err_dict_e2e():
    """
    end-to-end test for betti_num_err_dict
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
        betti_number_error_all_classes(
            gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
        )
    assert str(e_info.value) == "Invalid binary segmentation"

    # multiclass seg task will give the following:
    # GT has B0=1, B1=0, B2=0 for all 8 labels and combined
    #
    # Pred has the same Betti numbers for label 1, 3, 5, 7
    # label 2 middle 2x2x2 hollow:
    #        B0=1, B1=0, B2=1
    # label 4 missing:
    #        B0=0, B1=0, B2=0
    # label 6, 8 solid donut:
    #        B0=1, B1=1, B2=0
    # merged binary:
    #        B0=1, B1=1, B2=1
    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "Betti_0_error": 0},
        "2": {"label": "R-PCA", "Betti_0_error": 0},
        "3": {"label": "L-PCA", "Betti_0_error": 0},
        "4": {"label": "R-ICA", "Betti_0_error": 1},
        "5": {"label": "R-MCA", "Betti_0_error": 0},
        "6": {"label": "L-ICA", "Betti_0_error": 0},
        "7": {"label": "L-MCA", "Betti_0_error": 0},
        "8": {"label": "R-Pcom", "Betti_0_error": 0},
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 1 / 8},
        "CoW": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 1},
    }


def test_bettiError_nolabels_binary():
    """
    what if there is no labels in both gt and pred? -> B0Err=0 B1Err=0
    what if there is no labels in gt? -> depends on pred labels
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    assert betti_number_error_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.BINARY_SEGMENTATION
    ) == {"1": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0}}

    # gt is clean slate, but pred has some predictions
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION
    ) == {"1": {"label": "CoW", "Betti_0_error": 1, "Betti_1_error": 0}}


def test_bettiError_nolabels_multiclass():
    """
    same as test_bettiError_nolabels_binary but for multiclass

    what if there is no labels in both gt and pred? -> B0err_average=0, CoW=0
    what if there is no labels in gt? -> depends on pred labels
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    assert betti_number_error_all_classes(
        gt=gt_img, pred=gt_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 0},
        "CoW": {"label": "CoW", "Betti_0_error": 0, "Betti_1_error": 0},
    }

    # gt is clean slate, but pred has some predictions
    gt_path = "test_metrics/shape_6x3_2D.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    assert betti_number_error_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    ) == {
        "1": {"label": "BA", "Betti_0_error": 1},
        "B0err_average": {"label": "B0err_average", "Betti_0_error": 1},
        "CoW": {"label": "CoW", "Betti_0_error": 1, "Betti_1_error": 0},
    }
