"""
run the tests with pytest
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from topcow24_eval.constants import ANTERIOR_LABELS, POSTERIOR_LABELS
from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

from .topology_matching import compare_topo_dict, populate_topo_dict, topology_matching

##############################################################
#   ________________________________
# < 7. Tests for topology matching >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR = Path("test_assets/seg_metrics/topcow_roi")

#####################################################
# test populate_topo_dict()


def test_populate_topo_dict_anterior_for_gt():
    """
    populate_topo_dict() for ground-truth
    for anterior
    """
    # 2D array with labels (10, 11, 12, 15)
    array_2d = np.array(
        [
            [10, 10, 10, 0, 0, 11, 11, 11, 0, 0],
            [10, 10, 10, 0, 0, 11, 11, 11, 0, 0],
            [0, 0, 0, 12, 12, 12, 0, 0, 15, 15],
            [0, 0, 0, 12, 12, 12, 0, 0, 15, 15],
            [0, 0, 0, 12, 12, 12, 0, 0, 15, 15],
            [11, 11, 0, 0, 0, 12, 12, 12, 0, 0],
            [11, 11, 0, 0, 0, 12, 12, 12, 0, 0],
        ]
    )
    # convert to 3D
    mask_arr = np.expand_dims(array_2d, axis=-1)

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    topo_dict = {}

    populate_topo_dict(
        labels=ANTERIOR_LABELS,
        topo_dict=topo_dict,
        padded=padded,
    )

    assert topo_dict == {
        "Acom": {"b0": 1, "neighbors": [12]},
        "R-ACA": {"b0": 2, "neighbors": [12, 15]},
        "L-ACA": {"b0": 1, "neighbors": [10, 11, 15]},
        "3rd-A2": {"b0": 1, "neighbors": [11, 12]},
    }


def test_populate_topo_dict_posterior_for_pred():
    """
    populate_topo_dict() for prediction,
    thus gt:sitk.Image and pred:sitk.Image are supplied
    for posterior
    """
    # mask_2d with labels (2, 3, 8, 9)
    arr_2d = np.array(
        [
            [0, 2, 0, 2, 0, 0, 8, 8],
            [0, 2, 0, 2, 0, 8, 8, 8],
            [3, 3, 3, 0, 0, 8, 0, 0],
            [3, 0, 0, 9, 9, 9, 9, 0],
            [3, 3, 0, 0, 0, 0, 9, 9],
            [0, 3, 3, 3, 0, 9, 9, 0],
        ]
    )
    # convert to 3D
    mask_arr = np.expand_dims(arr_2d, axis=-1)

    # this is used for both gt:sitk.Image and pred:sitk.Image
    img = sitk.GetImageFromArray(mask_arr)

    #############################################
    topo_dict = {}

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    populate_topo_dict(
        labels=POSTERIOR_LABELS, topo_dict=topo_dict, padded=padded, gt=img, pred=img
    )

    assert topo_dict == {
        "R-PCA": {"detection": "TP", "b0": 2, "neighbors": [3]},
        "L-PCA": {"detection": "TP", "b0": 1, "neighbors": [2, 9]},
        "R-Pcom": {"detection": "TP", "b0": 1, "neighbors": [9]},
        "L-Pcom": {"detection": "TP", "b0": 1, "neighbors": [3, 8]},
        "LR_flipped": False,
    }


#####################################################
# test compare_topo_dict()


def test_compare_topo_dict():
    """
    two dummy topo_dicts from do not match to match
    """
    # [X] detection not matched
    gt_topo = {"BA": {}}
    pred_topo = {"BA": {"detection": "FP"}}
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is False
    gt_topo = {"BA": {}}
    pred_topo = {"BA": {"detection": "FN"}}
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is False

    # [X] neighbors not matched
    gt_topo = {"BA": {"neighbors": [2, 12]}}
    pred_topo = {"BA": {"detection": "TP", "neighbors": [11, 22, 33]}}
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is False

    # [X] b0 not matched
    gt_topo = {"BA": {"neighbors": [2, 12], "b0": 42}}
    pred_topo = {"BA": {"detection": "TP", "neighbors": [2, 12], "b0": 100}}
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is False

    # [X] LR flipped
    gt_topo = {"BA": {"neighbors": [2, 12], "b0": 42}}
    pred_topo = {
        "BA": {"detection": "TP", "neighbors": [2, 12], "b0": 42},
        "LR_flipped": True,
    }
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is False

    # finally all matched :)
    gt_topo = {"BA": {"neighbors": [2, 12], "b0": 42}}
    pred_topo = {
        "BA": {"detection": "TP", "neighbors": [2, 12], "b0": 42},
        "LR_flipped": False,
    }
    assert compare_topo_dict(gt_topo, pred_topo, (1,)) is True


#####################################################
# e2e test topology_matching()


def test_topology_matching_anterior_np_arr():
    """
    topology_matching(gt=gt, pred=pred) for anterior
    on two dummy numpy arrays with labels (4,6,10,11,12,15)
    """
    # labels 4, 6, 10, 11, 12, and 15
    gt_2d = np.array(
        [
            [4, 6, 10, 10, 12, 12],
            [4, 4, 6, 10, 15, 15],
            [11, 11, 6, 6, 12, 15],
            [11, 4, 11, 12, 12, 15],
        ]
    )
    gt = sitk.GetImageFromArray(np.expand_dims(gt_2d, axis=-1))

    # pred change the following:
    # label-11 increase b0+=1
    # label-12 remove 4 pixels, IoU=1/5 -> FN
    #  only connects to label-10 and label-15

    pred_2d = np.array(
        [
            [4, 6, 10, 10, 12, 0],
            [4, 4, 6, 10, 15, 15],
            [11, 0, 6, 6, 0, 15],
            [11, 4, 11, 0, 0, 15],
        ]
    )
    pred = sitk.GetImageFromArray(np.expand_dims(pred_2d, axis=-1))

    topo_dict = topology_matching(gt=gt, pred=pred)

    assert topo_dict == {
        "gt_topology": {
            "anterior": {
                "graph": [1, 1, 1, 1],
                "Acom": {"b0": 1, "neighbors": [4, 6, 12, 15]},
                "R-ACA": {"b0": 1, "neighbors": [4, 6, 12]},
                "L-ACA": {"b0": 2, "neighbors": [6, 10, 11, 15]},
                "3rd-A2": {"b0": 1, "neighbors": [6, 10, 12]},
            },
            "posterior": {
                "graph": [0, 0, 0, 0],
                "R-PCA": {"b0": 0, "neighbors": []},
                "L-PCA": {"b0": 0, "neighbors": []},
                "R-Pcom": {"b0": 0, "neighbors": []},
                "L-Pcom": {"b0": 0, "neighbors": []},
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [0, 1, 1, 1],
                "Acom": {"detection": "TP", "b0": 1, "neighbors": [4, 6, 12, 15]},
                "R-ACA": {"detection": "TP", "b0": 2, "neighbors": [4, 6]},
                "L-ACA": {"detection": "FN", "b0": 1, "neighbors": [10, 15]},
                "3rd-A2": {"detection": "TP", "b0": 1, "neighbors": [6, 10, 12]},
                "LR_flipped": False,
            },
            "posterior": {
                "graph": [0, 0, 0, 0],
                "R-PCA": {"detection": "TN", "b0": 0, "neighbors": []},
                "L-PCA": {"detection": "TN", "b0": 0, "neighbors": []},
                "R-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                "L-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                "LR_flipped": False,
            },
        },
        "match_verdict": {"anterior": False, "posterior": True},
    }


def test_topology_matching_posterior_np_arr():
    """
    topology_matching(gt=gt, pred=pred) for posterior
    on two dummy numpy arrays with labels (2,3,8,9)
    """
    # gt:sitk.Image is slightly different from pred
    # label-2 only 1 connected component (vs b0=2 in pred)
    # label-3 no longer connected to label-9
    # label-8 much smaller = 1/6 IoU -> FN
    gt_2d = np.array(
        [
            [0, 2, 2, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 0, 0],
            [3, 3, 3, 0, 0, 8, 0, 0],
            [3, 0, 0, 0, 9, 9, 9, 0],
        ]
    )
    gt = sitk.GetImageFromArray(np.expand_dims(gt_2d, axis=-1))

    # pred_2d with labels (2, 3, 8, 9)
    pred_2d = np.array(
        [
            [0, 2, 0, 2, 0, 0, 8, 8],
            [0, 2, 0, 2, 0, 8, 8, 8],
            [3, 3, 3, 0, 0, 8, 0, 0],
            [3, 0, 0, 9, 9, 9, 9, 0],
        ]
    )
    pred = sitk.GetImageFromArray(np.expand_dims(pred_2d, axis=-1))

    topo_dict = topology_matching(gt=gt, pred=pred)

    assert topo_dict == {
        "gt_topology": {
            "anterior": {
                "graph": [0, 0, 0, 0],
                "Acom": {"b0": 0, "neighbors": []},
                "R-ACA": {"b0": 0, "neighbors": []},
                "L-ACA": {"b0": 0, "neighbors": []},
                "3rd-A2": {"b0": 0, "neighbors": []},
            },
            "posterior": {
                "graph": [1, 0, 0, 1],
                "R-PCA": {"b0": 1, "neighbors": [3]},
                "L-PCA": {"b0": 1, "neighbors": [2]},
                "R-Pcom": {"b0": 1, "neighbors": [9]},
                "L-Pcom": {"b0": 1, "neighbors": [8]},
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [0, 0, 0, 0],
                "Acom": {"detection": "TN", "b0": 0, "neighbors": []},
                "R-ACA": {"detection": "TN", "b0": 0, "neighbors": []},
                "L-ACA": {"detection": "TN", "b0": 0, "neighbors": []},
                "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                "LR_flipped": False,
            },
            "posterior": {
                "graph": [1, 0, 0, 1],
                "R-PCA": {"detection": "TP", "b0": 2, "neighbors": [3]},
                "L-PCA": {"detection": "TP", "b0": 1, "neighbors": [2, 9]},
                "R-Pcom": {"detection": "FN", "b0": 1, "neighbors": [9]},
                "L-Pcom": {"detection": "TP", "b0": 1, "neighbors": [3, 8]},
                "LR_flipped": False,
            },
        },
        "match_verdict": {"anterior": True, "posterior": False},
    }


def test_topology_matching_antpos_np_arr():
    """
    np array from range(16) without 13 14
    """
    # gt diagonal all 0
    gt_2d = np.array(
        [
            [0, 1, 2, 3],
            [4, 0, 6, 7],
            [8, 9, 0, 11],
            [12, 0, 0, 0],
        ]
    )
    gt = sitk.GetImageFromArray(np.expand_dims(gt_2d, axis=-1))

    # pred_2d with all range(16)
    pred_2d = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 0, 0, 15],
        ]
    )
    pred = sitk.GetImageFromArray(np.expand_dims(pred_2d, axis=-1))

    topo_dict = topology_matching(gt=gt, pred=pred)

    assert topo_dict == {
        "gt_topology": {
            "anterior": {
                "graph": [0, 0, 0, 0],
                "Acom": {"b0": 0, "neighbors": []},
                "R-ACA": {"b0": 1, "neighbors": [6, 7]},
                "L-ACA": {"b0": 1, "neighbors": [8, 9]},
                "3rd-A2": {"b0": 0, "neighbors": []},
            },
            "posterior": {
                "graph": [1, 0, 1, 1],
                "R-PCA": {"b0": 1, "neighbors": [1, 3, 6, 7]},
                "L-PCA": {"b0": 1, "neighbors": [2, 6, 7]},
                "R-Pcom": {"b0": 1, "neighbors": [4, 9, 12]},
                "L-Pcom": {"b0": 1, "neighbors": [4, 6, 8, 12]},
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [0, 1, 1, 0],
                "Acom": {"detection": "FP", "b0": 1, "neighbors": [5, 6, 7, 9, 11, 15]},
                "R-ACA": {"detection": "TP", "b0": 1, "neighbors": [6, 7, 10, 15]},
                "L-ACA": {"detection": "TP", "b0": 1, "neighbors": [8, 9]},
                "3rd-A2": {"detection": "FP", "b0": 1, "neighbors": [10, 11]},
                "LR_flipped": False,
            },
            "posterior": {
                "graph": [1, 0, 1, 1],
                "R-PCA": {"detection": "TP", "b0": 1, "neighbors": [1, 3, 5, 6, 7]},
                "L-PCA": {"detection": "TP", "b0": 1, "neighbors": [2, 6, 7]},
                "R-Pcom": {"detection": "TP", "b0": 1, "neighbors": [4, 5, 9, 12]},
                "L-Pcom": {
                    "detection": "TP",
                    "b0": 1,
                    "neighbors": [4, 5, 6, 8, 10, 12],
                },
                "LR_flipped": False,
            },
        },
        "match_verdict": {"anterior": False, "posterior": False},
    }


def test_topology_matching_topcow003ct():
    """
    topcow ct 003 vs LPS_ACA_flipped.nii.gz

    Topcow 003
        has P1
        no L-A1

        "anterior": {
            "L-A1": 0,
            "Acom": 1,
            "3rd-A2": 0,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 1,
            "L-P1": 1,
            "R-P1": 1,
            "R-Pcom": 1,
        },

    LPS_ACA_flipped.nii.gz has ACAs flipped

    NOTE: topcow ct 003 and LPS_ACA_flipped have the same directions:

    topcow_ct_roi_003.nii.gz
    image.GetDirection() = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    LPS_ACA_flipped.nii.gz
    image.GetDirection() = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    """
    gt, _ = load_image_and_array_as_uint8(TESTDIR / "topcow_ct_roi_003.nii.gz", True)
    pred, _ = load_image_and_array_as_uint8(TESTDIR / "LPS_ACA_flipped.nii.gz", True)

    topo_dict = topology_matching(gt=gt, pred=pred)

    assert topo_dict == {
        "gt_topology": {
            "anterior": {
                "graph": [0, 1, 0, 1],
                "Acom": {"b0": 1, "neighbors": [11, 12]},
                "R-ACA": {"b0": 1, "neighbors": [4, 10]},
                "L-ACA": {"b0": 1, "neighbors": [10]},
                "3rd-A2": {"b0": 0, "neighbors": []},
            },
            "posterior": {
                "graph": [1, 1, 1, 1],
                "R-PCA": {"b0": 1, "neighbors": [1, 8]},
                "L-PCA": {"b0": 1, "neighbors": [1, 9]},
                "R-Pcom": {"b0": 1, "neighbors": [2, 4]},
                "L-Pcom": {"b0": 1, "neighbors": [3, 6]},
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [1, 0, 0, 0],
                "Acom": {"detection": "FN", "b0": 0, "neighbors": []},
                "R-ACA": {"detection": "FN", "b0": 1, "neighbors": [6, 12]},
                "L-ACA": {"detection": "FN", "b0": 2, "neighbors": [6, 11]},
                "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                "LR_flipped": True,
            },
            "posterior": {
                "graph": [0, 1, 1, 0],
                "R-PCA": {"detection": "FN", "b0": 1, "neighbors": [1, 3]},
                "L-PCA": {"detection": "FN", "b0": 1, "neighbors": [1, 2]},
                "R-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                "L-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                "LR_flipped": False,
            },
        },
        "match_verdict": {"anterior": False, "posterior": False},
    }


def test_topology_matching_topcow023mr():
    """
    topcow mr 023 vs LPS_ICA_PCA_flipped.nii.gz

    topcow 023, no L-A1 and no R-P1

        "anterior": {
            "L-A1": 0,
            "Acom": 1,
            "3rd-A2": 0,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 0,
            "L-P1": 1,
            "R-P1": 0,
            "R-Pcom": 1,
        },

    LPS_ICA_PCA_flipped.nii.gz has both ICAs and PCAs flipped

    NOTE: topcow mr 023 and LPS_ICA_PCA_flipped have tiny
    differences in their GetDirection():

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
    gt, _ = load_image_and_array_as_uint8(TESTDIR / "topcow_mr_roi_023.nii.gz", True)
    pred, _ = load_image_and_array_as_uint8(
        TESTDIR / "LPS_ICA_PCA_flipped.nii.gz", True
    )

    topo_dict = topology_matching(gt=gt, pred=pred)

    assert topo_dict == {
        "gt_topology": {
            "anterior": {
                "graph": [0, 1, 0, 1],
                "Acom": {"b0": 1, "neighbors": [11, 12]},
                "R-ACA": {"b0": 1, "neighbors": [4, 10]},
                "L-ACA": {"b0": 1, "neighbors": [10]},
                "3rd-A2": {"b0": 0, "neighbors": []},
            },
            "posterior": {
                "graph": [0, 1, 0, 1],
                "R-PCA": {"b0": 1, "neighbors": [8]},
                "L-PCA": {"b0": 1, "neighbors": [1]},
                "R-Pcom": {"b0": 1, "neighbors": [2, 4]},
                "L-Pcom": {"b0": 0, "neighbors": []},
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [0, 0, 0, 0],
                "Acom": {"detection": "FN", "b0": 0, "neighbors": []},
                "R-ACA": {"detection": "FN", "b0": 0, "neighbors": []},
                "L-ACA": {"detection": "FN", "b0": 0, "neighbors": []},
                "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                "LR_flipped": True,
            },
            "posterior": {
                "graph": [0, 1, 1, 0],
                "R-PCA": {"detection": "FN", "b0": 1, "neighbors": [1]},
                "L-PCA": {"detection": "FN", "b0": 1, "neighbors": [1]},
                "R-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                "L-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                "LR_flipped": True,
            },
        },
        "match_verdict": {"anterior": False, "posterior": False},
    }
