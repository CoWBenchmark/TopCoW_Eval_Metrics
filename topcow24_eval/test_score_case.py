from pathlib import Path
from pprint import pprint

import SimpleITK as sitk

from topcow24_eval.score_case_task_1_seg import score_case_task_1_seg
from topcow24_eval.score_case_task_2_box import score_case_task_2_box
from topcow24_eval.score_case_task_3_edg import (
    filter_out_distance,
    score_case_task_3_edg,
)

TESTDIR = Path("./test_assets")


def test_score_case_task_1_seg():
    """
    score_case_task_1_seg() should be the same as
    test_e2e_TopCoWEvaluation_Task_1_Seg_no_crop's
    "case": self._case_results.to_dict() part,
    except the pred_fname and gt_fname path fields
    """
    test_dict = {"UZH": "Best #1"}
    gt = sitk.ReadImage(
        TESTDIR / "task_1_seg_ground-truth" / "shape_8x8x8_3D_8Cubes_gt.nii.gz"
    )
    pred = sitk.ReadImage(
        TESTDIR / "task_1_seg_predictions" / "shape_8x8x8_3D_8Cubes_pred.mha"
    )

    # should only mutate test_dict
    score_case_task_1_seg(gt=gt, pred=pred, metrics_dict=test_dict)

    # test_dict object still can be mutated
    test_dict["NUS"] = 2022

    print("test_dict =", test_dict)

    # thus the original parts of the dict are kept
    assert test_dict == {
        "UZH": "Best #1",
        "Dice_BA": 1.0,
        "Dice_R-PCA": 0.9333333333333333,
        "Dice_L-PCA": 1.0,
        "Dice_R-ICA": 0,
        "Dice_R-MCA": 1.0,
        "Dice_L-ICA": 0.8571428571428571,
        "Dice_L-MCA": 1.0,
        "Dice_R-Pcom": 0.8571428571428571,
        "Dice_ClsAvgDice": 0.8309523809523809,
        "Dice_MergedBin": 0.8869565217391304,
        "clDice": 0.0,
        "B0err_BA": 0,
        "B0err_R-PCA": 0,
        "B0err_L-PCA": 0,
        "B0err_R-ICA": 1,
        "B0err_R-MCA": 0,
        "B0err_L-ICA": 0,
        "B0err_L-MCA": 0,
        "B0err_R-Pcom": 0,
        "B0err_ClsAvgB0err": 0.125,
        "B0err_MergedBin": 0,
        "HD_BA": 0.0,
        "HD95_BA": 0.0,
        "HD_R-PCA": 0.0,
        "HD95_R-PCA": 0.0,
        "HD_L-PCA": 0.0,
        "HD95_L-PCA": 0.0,
        "HD_R-ICA": 90,
        "HD95_R-ICA": 90,
        "HD_R-MCA": 0.0,
        "HD95_R-MCA": 0.0,
        "HD_L-ICA": 1.0,
        "HD95_L-ICA": 1.0,
        "HD_L-MCA": 0.0,
        "HD95_L-MCA": 0.0,
        "HD_R-Pcom": 1.0,
        "HD95_R-Pcom": 1.0,
        "HD95_ClsAvgHD95": 11.5,
        "HD_ClsAvgHD": 11.5,
        "HD_MergedBin": 4.0,
        "HD95_MergedBin": 2.0,
        "all_detection_dicts": {
            "8": {"label": "R-Pcom", "Detection": "TP"},
            "9": {"label": "L-Pcom", "Detection": "TN"},
            "10": {"label": "Acom", "Detection": "TN"},
            "15": {"label": "3rd-A2", "Detection": "TN"},
        },
        "all_graph_dicts": {
            "anterior": {"gt_graph": [0, 0, 0, 0], "pred_graph": [0, 0, 0, 0]},
            "posterior": {"gt_graph": [0, 1, 1, 1], "pred_graph": [0, 1, 1, 1]},
        },
        "all_topo_dicts": {
            "gt_topology": {
                "anterior": {
                    "graph": [0, 0, 0, 0],
                    "Acom": {"b0": 0, "neighbors": []},
                    "R-ACA": {"b0": 0, "neighbors": []},
                    "L-ACA": {"b0": 0, "neighbors": []},
                    "3rd-A2": {"b0": 0, "neighbors": []},
                },
                "posterior": {
                    "graph": [0, 1, 1, 1],
                    "R-PCA": {"b0": 1, "neighbors": [1, 3, 4, 5, 6, 7, 8]},
                    "L-PCA": {"b0": 1, "neighbors": [1, 2, 4, 5, 6, 7, 8]},
                    "R-Pcom": {"b0": 1, "neighbors": [1, 2, 3, 4, 5, 6, 7]},
                    "L-Pcom": {"b0": 0, "neighbors": []},
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
                    "graph": [0, 1, 1, 1],
                    "R-PCA": {
                        "detection": "TP",
                        "b0": 1,
                        "neighbors": [1, 3, 5, 6, 7, 8],
                    },
                    "L-PCA": {
                        "detection": "TP",
                        "b0": 1,
                        "neighbors": [1, 2, 5, 6, 7, 8],
                    },
                    "R-Pcom": {
                        "detection": "TP",
                        "b0": 1,
                        "neighbors": [1, 2, 3, 5, 6, 7],
                    },
                    "L-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": True,
                },
            },
            "match_verdict": {"anterior": True, "posterior": False},
        },
        "NUS": 2022,
    }


def test_score_case_task_2_box():
    """
    reuse test_iou_dict_from_files_hollow_6x3x3_4x7x3()
    """
    test_dict = {"UZH": "Best #1"}

    gt_path = (
        TESTDIR / "box_metrics/test_iou_dict_from_files_hollow_6x3x3_4x7x3_box_1.TXT"
    )
    pred_path = (
        TESTDIR / "box_metrics/test_iou_dict_from_files_hollow_6x3x3_4x7x3_box_2.JSON"
    )

    # should only mutate test_dict
    score_case_task_2_box(gt_path=gt_path, pred_path=pred_path, metrics_dict=test_dict)

    # test_dict object still can be mutated
    test_dict["NUS"] = 2022

    print("test_dict =")
    pprint(test_dict)

    # thus the original parts of the dict are kept
    assert test_dict == {
        "UZH": "Best #1",
        "Boundary_IoU": 0.3,  # 30 / 100,
        "IoU": 0.35294117647058826,  # 36 / 102,
        "NUS": 2022,
    }


def test_score_case_task_3_edg():
    """
    reuse test_graph_dict_from_files_yml()
    """

    test_dict = {"UZH": "Best #1"}

    gt_path = TESTDIR / "edg_metrics/topcow_ct_003.yml"
    pred_path = TESTDIR / "edg_metrics/antpos_np_arr_pred.json"

    # should only mutate test_dict
    score_case_task_3_edg(gt_path=gt_path, pred_path=pred_path, metrics_dict=test_dict)

    # test_dict object still can be mutated
    test_dict["NUS"] = 2022

    print("test_dict =")
    pprint(test_dict)

    # thus the original parts of the dict are kept
    assert test_dict == {
        "UZH": "Best #1",
        "all_graph_dicts": {
            "anterior": {
                "gt_graph": [0, 1, 0, 1],
                "pred_graph": [0, 1, 1, 0],
            },
            "posterior": {
                "gt_graph": [1, 1, 1, 1],
                "pred_graph": [1, 0, 1, 1],
            },
        },
        "anterior_distance": 1.4142135623730951,
        "posterior_distance": 1.0,
        "NUS": 2022,
    }


#############
# utility to filter out distance from graph_dict for task-3
def test_filter_out_distance():
    graph_dict = {
        "anterior": {
            "gt_graph": [0, 1, 0, 1],
            "pred_graph": [0, 1, 1, 0],
            "distance": 1.4142135623730951,
        },
        "posterior": {
            "gt_graph": [1, 1, 1, 1],
            "pred_graph": [1, 0, 1, 1],
            "distance": 1.0,
        },
    }

    graph_dict_wo_dist, anterior_distance, posterior_distance = filter_out_distance(
        graph_dict
    )

    assert graph_dict_wo_dist == {
        "anterior": {
            "gt_graph": [0, 1, 0, 1],
            "pred_graph": [0, 1, 1, 0],
            # "distance": 1.4142135623730951,
        },
        "posterior": {
            "gt_graph": [1, 1, 1, 1],
            "pred_graph": [1, 0, 1, 1],
            # "distance": 1.0,
        },
    }

    assert anterior_distance == 1.4142135623730951
    assert posterior_distance == 1.0

    # another test case
    graph_dict = {
        "anterior": {
            "Shark": 1234567,
            "Whale": [0],
            "distance": "???",
        },
        "posterior": {"distance": 42},
    }
    graph_dict_wo_dist, anterior_distance, posterior_distance = filter_out_distance(
        graph_dict
    )
    assert graph_dict_wo_dist == {
        "anterior": {
            "Shark": 1234567,
            "Whale": [0],
            # "distance": "???",
        },
        "posterior": {},
    }
    assert anterior_distance == "???"
    assert posterior_distance == 42
