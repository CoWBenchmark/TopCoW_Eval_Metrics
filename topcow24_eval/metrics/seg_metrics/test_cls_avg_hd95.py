"""
run the tests with pytest
"""

from pathlib import Path

import SimpleITK as sitk
from cls_avg_hd95 import hd95_all_classes, hd95_single_label
from topcow24_eval.constants import HD95_UPPER_BOUND, TASK
from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   ________________________________
# < 4. Tests for HD95 >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR_2D = Path("test_assets/seg_metrics/2D")
TESTDIR_3D = Path("test_assets/seg_metrics/3D")


def test_hd95_single_label_Fig5c():
    """
    example from Fig5c from Nat Method 2024
        Understanding metric-related pitfalls in
        image analysis validation

    also is the same example from Fig 63 of
    Common Limitations of Image Processing Metrics: A Picture Story

    HD = 11.31
    HD95 = 6.79
    """
    gt_path = TESTDIR_2D / "shape_14x14_Fig63_bin_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_14x14_Fig63_bin_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # HD = 11.31
    # HD95 = 6.79
    label = 1
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert round(hd100_score, 2) == 11.31
    assert round(hd95_score, 2) == 6.79


def test_hd95_single_label_Fig10():
    """
    example from Fig 10 of
    Common Limitations of Image Processing Metrics: A Picture Story

    Fig10 caption gives HD95 = 2.0
    """
    gt_path = TESTDIR_2D / "shape_11x11_2D_Fig10_bin_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_11x11_2D_Fig10_bin_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=1)
    assert hd95_score == 2.0
    assert hd100_score == 2.0


def test_hd95_single_label_Fig50_large_binary():
    """
    example from Fig 50 of
    Common Limitations of Image Processing Metrics: A Picture Story

    large structure from top row

    Fig50 caption gives HD95 = 0.0 for Pred 1
                        HD95 = 0.8 for Pred 2
    """
    gt_path = TESTDIR_2D / "shape_11x11_2D_Fig50_large_bin_gt.nii.gz"
    pred_1_path = TESTDIR_2D / "shape_11x11_2D_Fig50_large_bin_pred_1.nii.gz"
    pred_2_path = TESTDIR_2D / "shape_11x11_2D_Fig50_large_bin_pred_2.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_1_img, _ = load_image_and_array_as_uint8(pred_1_path)
    pred_2_img, _ = load_image_and_array_as_uint8(pred_2_path)

    # Pred 1 HD95 = 0.00
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_1_img, label=1)
    assert hd95_score == 0
    assert hd100_score == 1

    # Pred 2 HD95 = 0.8
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_2_img, label=1)
    # strangly HD95 calculated is 0.8499...?!
    # NOTE: "A Picture Story" seems to round down to 1 decimal
    assert hd95_score // 0.1 / 10 == 0.8499 // 0.1 / 10 == 0.8  # ???
    assert hd100_score == 1


def test_hd95_single_label_Fig50_small_multiclass():
    """
    example from Fig 50 of
    Common Limitations of Image Processing Metrics: A Picture Story

    small structure from bottom row

    Fig50 caption gives HD95 = 0.8 for Pred 1 (label-10)
                        HD95 = 1 for Pred 2   (label-15)
    """
    gt_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # for label-1, there is no mask for computation
    # trigger HD95_UPPER_BOUND
    label = 1
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd95_score == HD95_UPPER_BOUND == hd100_score

    # HD95 = 0.8 for Pred 1 (label-10)
    label = 10
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    # NOTE: "A Picture Story" seems to round down to 1 decimal
    # my calculation shows 0.8999???
    assert hd95_score // 0.1 / 10 == 0.899 // 0.1 / 10 == 0.8  # ???
    assert hd100_score == 1

    # HD95 = 1 for Pred 2   (label-15)
    label = 15
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd95_score == 1
    assert hd100_score == 1


def test_hd95_single_label_Fig54_5class():
    """
    example from Fig 54 of
    Common Limitations of Image Processing Metrics: A Picture Story

    5 planes for Pred 1-5 with label 1-5
    """
    gt_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    # Pred 1 HD=1.4, HD95=1.3
    label = 1
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd100_score // 0.1 / 10 == 1.4
    # BUT... why is hd95 1.3?? my calculation shows 1.02...
    assert hd95_score // 0.01 / 100 == 1.02  # ??? != 1.3!

    # Pred 2 HD=3.6, HD95=3.1
    label = 2
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd100_score // 0.1 / 10 == 3.6
    assert hd95_score // 0.1 / 10 == 3.1

    # Pred 3 HD=3, HD95=2
    label = 3
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd100_score == 3
    assert hd95_score == 2

    # Pred 4 HD=2.2, HD95=2
    label = 4
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    # BUT WHY??? my calculation shows 1.4 and 1.4 (sqrt of 2)
    assert hd100_score // 0.1 / 10 == 1.4  # ???
    assert hd95_score // 0.1 / 10 == 1.4  # ???

    # Pred 5 HD=2, HD95=1.2
    label = 5
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd100_score == 2
    # BUT my calculation shows hd95 to be 1.05?!
    assert hd95_score // 0.01 / 100 == 1.05  # ??? !=1.2!


def test_hd95_single_label_Fig59_hole():
    """
    example from Fig 59 of
    Common Limitations of Image Processing Metrics: A Picture Story

    with hole inside, Pred 1 smaller hole, Pred 2 bigger hole
    """
    gt_path = TESTDIR_2D / "shape_7x7_Fig59_bin_gt.nii.gz"
    pred_1_path = TESTDIR_2D / "shape_7x7_Fig59_bin_pred_1top.nii.gz"
    pred_2_path = TESTDIR_2D / "shape_7x7_Fig59_bin_pred_2top.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_1_img, _ = load_image_and_array_as_uint8(pred_1_path)
    pred_2_img, _ = load_image_and_array_as_uint8(pred_2_path)

    # Pred 1 HD = 1, HD95 = 1
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_1_img, label=1)
    assert hd95_score == 1
    assert hd100_score == 1

    # Pred 2 HD = 0, HD95 = 0
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_2_img, label=1)
    assert hd95_score == 0
    assert hd100_score == 0


def test_hd95_single_label_FPFN():
    """
    in case of FP or FN trigger worst value penalty
    """
    # mimic no labels in both gt and pred by reusing a clean slate
    gt_path = TESTDIR_2D / "shape_6x3_2D.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)

    assert hd95_single_label(gt=gt_img, pred=gt_img, label=1) == [
        HD95_UPPER_BOUND,
        HD95_UPPER_BOUND,
    ]

    # NOTE: in fact even for a mask with valid labels,
    # as long as the filtered img is empty in img == label
    # will trigger FP FN
    img, _ = load_image_and_array_as_uint8(
        TESTDIR_3D / "shape_2x2x2_3D_Fig50_label15_10_gt.nii.gz"
    )
    # non-existing labels will trigger FP FN
    label = 42
    assert hd95_single_label(
        gt=img,
        pred=img,
        label=label,
    ) == [
        HD95_UPPER_BOUND,
        HD95_UPPER_BOUND,
    ]
    label = 13
    assert hd95_single_label(
        gt=img,
        pred=img,
        label=label,
    ) == [
        HD95_UPPER_BOUND,
        HD95_UPPER_BOUND,
    ]


def test_hd95_single_label_3D_voxel_spacing():
    """
    with simple 3x3x3 sitk.Image, test that
    hd95 calculation uses spacing information
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # set the spacing to be a Pythagorean triple
    image1.SetSpacing((3, 4, 5))
    image2.SetSpacing((3, 4, 5))

    # label-1 in image1 is x-index 2
    #         in image2 is x-index 0
    # distance is 2*spacing-x = 2*3 = 6
    image1[2, 0, 0] = 1
    image2[0, 0, 0] = 1
    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))
    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    hd95_score, hd100_score = hd95_single_label(gt=image1, pred=image2, label=1)
    assert hd100_score == 6 == hd95_score

    # label-2 in image1 is at xy (0,1)
    #         in image2 is at xy (1,0)
    # distance is Pythagorean 5
    image1[0, 1, 0] = 2
    image2[1, 0, 0] = 2
    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))
    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    hd95_score, hd100_score = hd95_single_label(gt=image1, pred=image2, label=2)
    assert hd100_score == 5 == hd95_score

    # label-3 in image1 is at z-index 1
    #         in image2 is at z-index 2
    # distance is spacing-z = 5
    image1[:, :, 1] = 3
    image2[:, :, 2] = 3
    print("image1:")
    print(sitk.GetArrayViewFromImage(image1))
    print("image2:")
    print(sitk.GetArrayViewFromImage(image2))

    hd95_score, hd100_score = hd95_single_label(gt=image1, pred=image2, label=3)
    assert hd100_score == 5 == hd95_score


def test_hd95_single_label_3D_nipy_scaled_image():
    """
    use the scaled_image nifti example from
    https://nipy.org/nibabel/nifti_images.html

    verify hd95_single_label uses voxelspacing

    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])

    Size-X: 2, Spacing-X: 1
    Size-Y: 3, Spacing-Y: 2
    Size-Z: 4, Spacing-Z: 3
    """
    gt_path = TESTDIR_3D / "scaled_image_label_123_gt.nii.gz"
    pred_path = TESTDIR_3D / "scaled_image_label_123_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path, log_sitk_attr=True)
    pred_img, _ = load_image_and_array_as_uint8(pred_path, log_sitk_attr=True)

    # Label-1 GT and Pred differ by a corner voxel along the Y-size
    # the distance is thus Spacing-Y * 2 = 2 * 2 = 4
    # 95th percentile of [4.0, 0, 0, 0, 0] gives 3.2
    label = 1
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert hd100_score == 4
    assert round(hd95_score, 1) == 3.2

    # Label-2 GT and Pred differ by
    # diagonal voxel on the X-Y plane
    # = sqrt(Spacing-Y**2 + Spacing-X**2) =sqrt(4+1) = 2.236
    label = 2
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert round(hd100_score, 3) == 2.236  # sqrt(5)
    # 95th percentile of [2.236068, 2.236068, 2.236068] is still 2.236
    assert round(hd95_score, 3) == 2.236  # sqrt(5)

    # Label-3 GT and Pred differ by
    # one closest by 1 Z size = 3
    # another diagonal in 3D = sqrt(1+4+9) = sqrt(14) = 3.742
    # thus HD = max(3, 3.742) = 3.742
    # 95th percentile of [3.7416575, 3.0] is 3.7
    label = 3
    hd95_score, hd100_score = hd95_single_label(gt=gt_img, pred=pred_img, label=label)
    assert round(hd100_score, 3) == 3.742
    assert round(hd95_score, 2) == 3.7


def test_hd95_single_label_itk_tutorial():
    """
    use the same examples as ITK tutorial:
    http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html

    Segmentation-Representation-and-the-Hausdorff-Distance

    surface_hausdorff_distance()
    Surface Hausdorff result (reference1-segmentation): 6.0
    Surface Hausdorff result (reference2-segmentation): 8.9442720413208
    """
    # Create our segmentations and display
    image_size = [64, 64]
    circle_center = [30, 30]
    circle_radius = [20, 20]

    # A filled circle with radius R
    seg = (
        sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center)
        > 200
    )
    # A torus with inner radius r
    reference_segmentation1 = seg - (
        sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center)
        > 240
    )
    # A torus with inner radius r_2<r
    reference_segmentation2 = seg - (
        sitk.GaussianSource(sitk.sitkUInt8, image_size, circle_radius, circle_center)
        > 250
    )

    # joinSeries to stack to 3D
    seg_3d = sitk.JoinSeries(seg)
    reference_segmentation1_3d = sitk.JoinSeries(reference_segmentation1)
    reference_segmentation2_3d = sitk.JoinSeries(reference_segmentation2)

    # Use reference1, larger inner annulus radius, the surface based computation
    # Surface Hausdorff result (reference1-segmentation): 6.0
    _, hd100_score = hd95_single_label(
        gt=reference_segmentation1_3d, pred=seg_3d, label=1
    )
    assert hd100_score == 6.0

    # Use reference2, smaller inner annulus radius, the surface based computation
    # Surface Hausdorff result (reference2-segmentation): 8.9442720413208
    _, hd100_score = hd95_single_label(
        gt=reference_segmentation2_3d, pred=seg_3d, label=1
    )
    assert hd100_score == 8.9442720413208


#######################################################################
# e2e test for hd_dict from hd95_all_classes()


def test_hd95_all_classes_Fig5c():
    """
    example from Fig5c from Nat Method 2024
        Understanding metric-related pitfalls in
        image analysis validation

    also is the same example from Fig 63 of
    Common Limitations of Image Processing Metrics: A Picture Story

    HD = 11.31
    HD95 = 6.79
    """
    gt_path = TESTDIR_2D / "shape_14x14_Fig63_bin_gt.nii.gz"
    pred_path = TESTDIR_2D / "shape_14x14_Fig63_bin_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    hd_dict = hd95_all_classes(gt=gt_img, pred=pred_img, task=TASK.BINARY_SEGMENTATION)

    assert hd_dict == {
        "1": {"label": "MergedBin", "HD95": 6.788224983215328, "HD": 11.313708305358887}
    }


def test_hd95_all_classes_Fig54_5class():
    """
    example from Fig 54 of
    Common Limitations of Image Processing Metrics: A Picture Story

    5 planes for Pred 1-5 with label 1-5
    """
    gt_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_gt.nii.gz"
    pred_path = TESTDIR_3D / "shape_11x11x5_3D_Fig54_label1-5_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path)
    pred_img, _ = load_image_and_array_as_uint8(pred_path)

    hd_dict = hd95_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert hd_dict == {
        # Pred 1 HD=1.4, HD95=1.3
        # BUT... why is hd95 1.3?? my calculation shows 1.02...
        "1": {"label": "BA", "HD95": 1.0207106769084933, "HD": 1.4142135381698608},
        # Pred 2 HD=3.6, HD95=3.1
        "2": {"label": "R-PCA", "HD95": 3.184441375732422, "HD": 3.605551242828369},
        # Pred 3 HD=3, HD95=2
        "3": {"label": "L-PCA", "HD95": 2.0, "HD": 3.0},
        # Pred 4 HD=2.2, HD95=2
        # BUT WHY??? my calculation shows 1.4 and 1.4 (sqrt of 2)
        "4": {"label": "R-ICA", "HD95": 1.4142135381698608, "HD": 1.4142135381698608},
        # Pred 5 HD=2, HD95=1.2
        # BUT my calculation shows hd95 to be 1.05?!
        "5": {"label": "R-MCA", "HD95": 1.0500000000000007, "HD": 2.0},
        # avg of HD95 = 1.73
        "ClsAvgHD95": {"label": "ClsAvgHD95", "HD95": 1.7338731181621554},
        # avg of HD = 2.28
        "ClsAvgHD": {"label": "ClsAvgHD", "HD": 2.286795663833618},
        "MergedBin": {"label": "MergedBin", "HD95": 3.0, "HD": 3.605551242828369},
    }


def test_hd95_all_classes_3D_nipy_scaled_image():
    """
    use the scaled_image nifti example from
    https://nipy.org/nibabel/nifti_images.html

    verify hd95_single_label uses voxelspacing

    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])

    Size-X: 2, Spacing-X: 1
    Size-Y: 3, Spacing-Y: 2
    Size-Z: 4, Spacing-Z: 3
    """
    gt_path = TESTDIR_3D / "scaled_image_label_123_gt.nii.gz"
    pred_path = TESTDIR_3D / "scaled_image_label_123_pred.nii.gz"

    gt_img, _ = load_image_and_array_as_uint8(gt_path, log_sitk_attr=True)
    pred_img, _ = load_image_and_array_as_uint8(pred_path, log_sitk_attr=True)

    hd_dict = hd95_all_classes(
        gt=gt_img, pred=pred_img, task=TASK.MULTICLASS_SEGMENTATION
    )

    assert hd_dict == {
        # Label-1 GT and Pred differ by a corner voxel along the Y-size
        # the distance is thus Spacing-Y * 2 = 2 * 2 = 4
        # 95th percentile of [4.0, 0, 0, 0, 0] gives 3.2
        "1": {"label": "BA", "HD95": 3.1999999999999993, "HD": 4.0},
        # Label-2 GT and Pred differ by
        # diagonal voxel on the X-Y plane
        # = sqrt(Spacing-Y**2 + Spacing-X**2) =sqrt(4+1) = 2.236
        # 95th percentile of [2.236068, 2.236068, 2.236068] is still 2.236
        "2": {"label": "R-PCA", "HD95": 2.2360680103302, "HD": 2.2360680103302},
        # Label-3 GT and Pred differ by
        # one closest by 1 Z size = 3
        # another diagonal in 3D = sqrt(1+4+9) = sqrt(14) = 3.742
        # thus HD = max(3, 3.742) = 3.742
        # 95th percentile of [3.7416575, 3.0] is 3.7
        "3": {"label": "L-PCA", "HD95": 3.7045746207237245, "HD": 3.7416574954986572},
        # avg of 3.2, 2.236, 3.7 = 3.046
        "ClsAvgHD95": {"label": "ClsAvgHD95", "HD95": 3.0468808770179745},
        # avg of 4, 2.236, 3.74 = 3.325
        "ClsAvgHD": {"label": "ClsAvgHD", "HD": 3.3259085019429526},
        "MergedBin": {"label": "MergedBin", "HD95": 2.69442720413208, "HD": 3.0},
    }
