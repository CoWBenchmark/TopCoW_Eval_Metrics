"""
run the tests with pytest
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from cls_avg_b0 import (
    betti_number_error_all_classes,
    betti_number_error_single_label,
    connected_components,
)
from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

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

TESTDIR_2D = Path("test_assets/seg_metrics/2D")
TESTDIR_3D = Path("test_assets/seg_metrics/3D")


def test_no_betti():
    """
    all zeros should give Betti number: 0
    """
    b0, _, sizes = connected_components(np.zeros((42, 42, 42)))
    assert b0 == 0
    assert sizes == []


def test_one_betti():
    """
    all ones should give Betti number: 1
    """
    b0, _, sizes = connected_components(np.ones((42, 42, 42)))
    assert b0 == 1
    assert sizes == [42**3]


def test_betti_number_skimage_euler_eg():
    """
    based on https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_euler_number.html
    """
    # Define a volume of 7x7x7 voxels
    n = 7
    cube = np.ones((n, n, n), dtype=bool)
    # Add a tunnel
    c = int(n / 2)
    cube[c, :, c] = False
    # Add a new hole
    cube[int(3 * n / 4), c - 1, c - 1] = False
    # Add a hole in neighborhood of previous one
    cube[int(3 * n / 4), c, c] = False
    # Add a second tunnel
    cube[:, c, int(3 * n / 4)] = False

    # invert the cube foreground
    cube = np.invert(cube)

    b0, _, sizes = connected_components(cube)

    assert b0 == 3
    assert sizes == [2, 7, 7]  # sorted


def test_singleBlob_noBorder():
    """
    GT is a 2x2x2 blob all filled with label 1 and no border (8 cubes)
    Pred is the same blob but missing a corner voxel (7 cubes)
    """
    gt_path = TESTDIR_3D / "betti_num_singleBlob_2x2x2_gt.nii.gz"
    pred_path = TESTDIR_3D / "betti_num_singleBlob_2x2x2_pred.nii.gz"

    _, gt_mask = load_image_and_array_as_uint8(gt_path)
    _, pred_mask = load_image_and_array_as_uint8(pred_path)

    gt_b0, _, sizes = connected_components(gt_mask)
    assert gt_b0 == 1
    assert sizes == [8]

    pred_b0, _, sizes = connected_components(pred_mask)
    assert pred_b0 == 1
    assert sizes == [7]


def test_betti_number_error_single_label():
    """
    test betti_number integration to betti_number_error_single_label
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # image1 and image2 diagonal elements are either
    # all ones for image1 or 1,2,3 for image2
    for i in range(3):
        image1[i, i, i] = 1
        image2[i, i, i] = i + 1

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    label_1_b0err = betti_number_error_single_label(
        gt=image1,
        pred=image2,
        label=1,
    )
    label_2_b0err = betti_number_error_single_label(
        gt=image1,
        pred=image2,
        label=2,
    )
    label_3_b0err = betti_number_error_single_label(
        gt=image1,
        pred=image2,
        label=3,
    )

    assert label_1_b0err == 0  # no error for label-1
    assert label_2_b0err == 1  # 1 error for label-2
    assert label_3_b0err == 1  # 1 error for label-3


def test_betti_number_error_single_label_blank_slices():
    """
    test_betti_number_error_single_label but with blank slices
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)

    # slice 0 image1 blank, image2 has one label-7
    image2[:, :, 0] = 7
    # slice 1 image1 has two label-8, image2 blank
    image1[2, 0, 1] = 8
    image1[0, 2, 1] = 8
    # slice 2 both blank

    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))

    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    label_7_b0err = betti_number_error_single_label(
        gt=image1,
        pred=image2,
        label=7,
    )
    label_8_b0err = betti_number_error_single_label(
        gt=image1,
        pred=image2,
        label=8,
    )

    assert label_7_b0err == 1  # 1 error for label-7
    assert label_8_b0err == 2  # 2 error for label-8


def test_cls_avg_b0_RGB():
    """
    5x5 2D multiclass gt is three columns of label(1,2,3)
    pred is three columns of label(1,3,4)

    label-1 GT=1 Pred=1
    label-2 GT=1 Pred=0
    label-3 GT=1 Pred=1
    label-4 Gt=0 Pred=1

    MergedBin GT=1 Pred=3 (pred three separate stripes)
    """
    gt_path = TESTDIR_2D / "shape_5x5_2D_RGB_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_5x5_2D_RBY_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 0},
        "2": {"label": "R-PCA", "B0err": 1},
        "3": {"label": "L-PCA", "B0err": 0},
        "4": {"label": "R-ICA", "B0err": 1},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 0.5},
        "MergedBin": {"label": "MergedBin", "B0err": 2},
    }


def test_singleBlob_noBorder_betti_num_err_dict():
    """
    end to end test for betti_num_err_dict based on test_singleBlob_noBorder()
    """
    gt_path = TESTDIR_3D / "betti_num_singleBlob_2x2x2_gt.nii.gz"
    pred_path = TESTDIR_3D / "betti_num_singleBlob_2x2x2_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # label-1 for multiclass segmentation task is BA
    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 0},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 0.0},
        "MergedBin": {"label": "MergedBin", "B0err": 0},
    }


def test_twoIslands():
    """
    two islands of label 1
    Answer: B0=2 B1=0 B2=0
    """
    # touching the edge, i.e. no border
    mask_path = TESTDIR_3D / "shape_3x4x2_3D_twoIslands.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    b0, _, sizes = connected_components(mask)

    assert b0 == 2
    assert sizes == [3, 7]

    # padded manually
    mask_path = TESTDIR_3D / "shape_5x6x4_3D_twoIslands_padded.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    b0, _, sizes = connected_components(mask)

    assert b0 == 2
    assert sizes == [3, 7]  # padding does not affect b0cc


def test_multi_class_twoIslands():
    """
    similar to test_twoIslands() but the pred has two labels
    """
    gt_path = TESTDIR_3D / "shape_3x4x2_3D_twoIslands.nii.gz"
    pred_path = TESTDIR_3D / "shape_3x4x2_3D_twoIslands_multiclass.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # GT has B0=2, B1=0, B2=0 and only has label-1
    #
    # Pred has the same Betti numbers for:
    # label 1 and label 2 and merged binary

    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 0},
        "2": {"label": "R-PCA", "B0err": 2},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 1},
        "MergedBin": {"label": "MergedBin", "B0err": 0},
    }


def test_donut():
    """
    a solid donut
    Answer: B0 = 1, B1 = 1, B2 = 0
    """
    mask_path = TESTDIR_3D / "shape_5x7x9_3D_1donut.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    b0, _, sizes = connected_components(mask)

    assert b0 == 1
    assert sizes == [10 * 3]


def test_multi_class_donut():
    """
    similar to test_donut() but have pred has two labels
    """
    gt_path = TESTDIR_3D / "shape_5x7x9_3D_1donut.nii.gz"
    pred_path = TESTDIR_3D / "shape_5x7x9_3D_1donut_multiclass.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # GT has B0=1, B1=1, B2=0 and only has label-1
    #
    # Pred has the same Betti numbers for merged binary
    # but label-1 and label-6 should have B0=1, B1=0, B2=0

    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 0},
        "6": {"label": "L-ICA", "B0err": 1},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 1 / 2},
        "MergedBin": {"label": "MergedBin", "B0err": 0},
    }


def test_cross():
    """
    cross with a hole in the middle, sort of like a solid donut
    Answer same as solid donut
    """
    # touching the edge, i.e. no border
    mask_path = TESTDIR_3D / "shape_3x4x2_3D_cross.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    b0, _, sizes = connected_components(mask)

    assert b0 == 1
    assert sizes == [6 + 5]

    # padded manually
    mask_path = TESTDIR_3D / "shape_5x6x4_3D_cross_padded.nii.gz"
    _, mask = load_image_and_array_as_uint8(mask_path)
    b0, _, sizes = connected_components(mask)

    assert b0 == 1
    assert sizes == [6 + 5]


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
    gt_path = TESTDIR_3D / "shape_8x8x8_3D_8Cubes_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_8x8x8_3D_8Cubes_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

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
    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 0},
        "2": {"label": "R-PCA", "B0err": 0},
        "3": {"label": "L-PCA", "B0err": 0},
        "4": {"label": "R-ICA", "B0err": 1},
        "5": {"label": "R-MCA", "B0err": 0},
        "6": {"label": "L-ICA", "B0err": 0},
        "7": {"label": "L-MCA", "B0err": 0},
        "8": {"label": "R-Pcom", "B0err": 0},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 1 / 8},
        "MergedBin": {"label": "MergedBin", "B0err": 0},
    }


def test_bettiError_nolabels_multiclass():
    """
    what if there is no labels in both gt and pred? -> B0err_average=0, CoW=0
    what if there is no labels in gt? -> depends on pred labels
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    assert betti_number_error_all_classes(gt=gt_img, pred=gt_img) == {
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 0},
        "MergedBin": {"label": "MergedBin", "B0err": 0},
    }

    # gt is clean slate, but pred has some predictions
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"
    pred_path = TESTDIR_2D / "shape_6x3_2D_clDice_elong_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    assert betti_number_error_all_classes(gt=gt_img, pred=pred_img) == {
        "1": {"label": "BA", "B0err": 1},
        "ClsAvgB0err": {"label": "ClsAvgB0err", "B0err": 1},
        "MergedBin": {"label": "MergedBin", "B0err": 1},
    }
