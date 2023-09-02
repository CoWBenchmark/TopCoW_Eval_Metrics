"""
run the tests with pytest
"""
import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d

from metric_functions import cl_score, clDice, convert_multiclass_to_binary
from utils_nii_mha_sitk import load_image_and_array_as_uint8

##############################################################
#   ________________________________
# < 2. Tests for cl_score and clDice >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################


def test_cl_score_skeletonize_ellipse():
    """
    from skimage example on skeletonize ellipse
    https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize

    compare the ellipse with a vertical rod and a horizontal rod
    """
    X, Y = np.ogrid[0:9, 0:9]
    ellipse = (1.0 / 3 * (X - 4) ** 2 + (Y - 4) ** 2 < 3**2).astype(np.uint8)
    # ellipse is:
    #  ([[0, 0, 0, 1, 1, 1, 0, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 1, 1, 1, 1, 1, 0, 0],
    #    [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    # and the skeletonize(ellipse) is:
    #  ([[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #    [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    ########################################
    # compare ellipse with a vertical rod
    ########################################

    v_rod = np.zeros((9, 9), dtype=np.uint8)
    v_rod[:, 4] = 1
    #   ([[0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=uint8)

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(ellipse), v_image=v_rod)
    assert tprec == (4 / 4)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(v_rod), v_image=ellipse)
    assert tsens == (9 / 9)

    ########################################
    # compare ellipse with a horizontal rod
    ########################################
    h_rod = np.zeros((9, 9), dtype=np.uint8)
    h_rod[4, :] = 1
    # array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=skeletonize(ellipse), v_image=h_rod)
    assert tprec == (1 / 4)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=skeletonize(h_rod), v_image=ellipse)
    assert tsens == (5 / 9)


def test_cl_score_2D_blob():
    """
    6x3 2D with an elongated blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = "test_metrics/shape_6x3_2D_clDice_elong_gt.nii.gz"
    pred_path = "test_metrics/shape_6x3_2D_clDice_elong_pred.nii.gz"

    _, gt_mask = load_image_and_array_as_uint8(gt_path)
    _, pred_mask = load_image_and_array_as_uint8(pred_path)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (6 / 6)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (4 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_mask, v_l_gt=gt_mask) == 1


def test_cl_score_2D_Tshaped():
    """
    5x5 2D with a T-shaped blob gt and a vertical columnn pred
    this test for cl_score (topology precision & topology sensitivity)
    """
    gt_path = "test_metrics/shape_5x5_2D_clDice_Tshaped_gt.nii.gz"
    pred_path = "test_metrics/shape_5x5_2D_clDice_Tshaped_pred.nii.gz"

    _, gt_mask = load_image_and_array_as_uint8(gt_path)
    _, pred_mask = load_image_and_array_as_uint8(pred_path)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    assert tprec == (5 / 5)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)
    assert tsens == (3 / 4)

    # clDice = 2 * tprec * tsens / (tprec + tsens)
    assert clDice(v_p_pred=pred_mask, v_l_gt=gt_mask) == (3 / 2) / (7 / 4)
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

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    multiclass_pred_mask = convert_multiclass_to_binary(multiclass_pred_arr)
    multiclass_gt_mask = convert_multiclass_to_binary(multiclass_gt_arr)

    # test_cl_score_2D_Tshaped should match test_cl_score_2D_Tshaped with multiclass!
    assert (
        tprec
        == cl_score(
            s_skeleton=call_skeletonize(multiclass_pred_mask),
            v_image=multiclass_gt_mask,
        )
        == (5 / 5)
    )

    assert (
        tsens
        == cl_score(
            s_skeleton=call_skeletonize(multiclass_gt_mask),
            v_image=multiclass_pred_mask,
        )
        == (3 / 4)
    )

    assert (
        clDice(v_p_pred=pred_mask, v_l_gt=gt_mask)
        == clDice(v_p_pred=multiclass_pred_arr, v_l_gt=multiclass_gt_arr)
        == ((3 / 2) / (7 / 4))
    )
    # ~= 0.85714
