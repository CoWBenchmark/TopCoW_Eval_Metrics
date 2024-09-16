import math
from pathlib import Path

from graph_dict_from_files import graph_dict_from_files

TESTDIR = Path("test_assets/edg_metrics")


def test_graph_dict_from_files_yml():
    """
    get the graph_dict from yml files

    topcow_ct_003.yml
        anterior:
            L-A1:    0
            Acom:    1
            3rd-A2:  0
            R-A1:    1
        posterior:
            L-Pcom:  1
            L-P1:    1
            R-P1:    1
            R-Pcom:  1

    antpos_np_arr_pred.json
        anterior:
            L-A1:    0
            Acom:    1
            3rd-A2:  1
            R-A1:    0
        posterior:
            L-Pcom:  1
            L-P1:    0
            R-P1:    1
            R-Pcom:  1
    """
    gt_edg_path = TESTDIR / "topcow_ct_003.yml"
    pred_edg_path = TESTDIR / "antpos_np_arr_pred.json"

    graph_dict = graph_dict_from_files(gt_edg_path, pred_edg_path)

    # have Euclidean distance added
    assert graph_dict == {
        "anterior": {
            "gt_graph": [0, 1, 0, 1],
            "pred_graph": [0, 1, 1, 0],
            "distance": math.sqrt(2),
        },
        "posterior": {
            "gt_graph": [1, 1, 1, 1],
            "pred_graph": [1, 0, 1, 1],
            "distance": math.sqrt(1),
        },
    }


def test_graph_dict_from_files_json():
    """
    get the graph_dict from json files

    cow-ant-post-classification_gt.JSON
    {
        "anterior": {"L-A1": 1, "Acom": 1, "3rd-A2": 0, "R-A1": 1},
        "posterior": {"L-Pcom": 1, "L-P1": 0, "R-P1": 1, "R-Pcom": 0},
    }

    cow-ant-post-classification_pred.yml
    {
        "anterior": {"L-A1": 1, "Acom": 1, "3rd-A2": 1, "R-A1": 1},
        "posterior": {"L-Pcom": 0, "L-P1": 0, "R-P1": 0, "R-Pcom": 0},
    }
    """
    gt_edg_path = TESTDIR / "cow-ant-post-classification_gt.JSON"
    pred_edg_path = TESTDIR / "cow-ant-post-classification_pred.yml"

    graph_dict = graph_dict_from_files(gt_edg_path, pred_edg_path)

    # have Euclidean distance added
    assert graph_dict == {
        "anterior": {
            "gt_graph": [1, 1, 0, 1],
            "pred_graph": [1, 1, 1, 1],
            "distance": math.sqrt(1),
        },
        "posterior": {
            "gt_graph": [1, 0, 1, 0],
            "pred_graph": [0, 0, 0, 0],
            "distance": math.sqrt(2),
        },
    }
