"""
for Variant-balanced graph classification accuracy
(accuracy calculated during aggregate stage)

And

Distance between ground-truth and predicted 4-element vectors

Metrics for Task-3-CoW-Classification
"""

import os
import pprint
from pathlib import Path

from edge_dict_to_list import edge_dict_to_list
from scipy.spatial.distance import euclidean
from topcow24_eval.utils.utils_edge import parse_edge_json, parse_edge_yml


def graph_dict_from_files(
    gt_edg_path: str | os.PathLike, pred_edg_path: str | os.PathLike
) -> dict:
    """
    Generate a CoW graph dict from edge lists for gt and pred.
    Similar to seg_metrics/graph_classification/graph_classification.py
    But with additional Euclidean distance between gt and pred vectors.

    NOTE: the edge-list files should be .yml or .json

    NOTE: use scipy.spatial.distance.euclidean
    to get the Euclidean distance between vectors gt and pred
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html

    Inputs:
        gt_edg_path:
            path of the ground-truch edge list, {.json|.yml}
        pred_edg_path:
            path of the predicted edge list, {.json|.yml}

    Returns:
        graph_dict with
            1 = presence of edge
            0 = absence of edge
        {
            "anterior":
                {
                    "gt_graph": [1, 0, 0, 1],
                    "pred_graph": [1, 0, 1, 0],
                    "distance": 1.4142135623730951,
                },
            "posterior":
                {
                    "gt_graph": [0, 0, 1, 1],
                    "pred_graph": [1, 0, 1, 1],
                    "distance": 1.0,
                },
        }
    """
    print("\nClassify CoW Graph from Edge List")

    gt_edg_path = Path(gt_edg_path)
    pred_edg_path = Path(pred_edg_path)

    assert gt_edg_path.is_file(), "gt_edg_path doesn't exist!"
    assert pred_edg_path.is_file(), "pred_edg_path doesn't exist!"

    if gt_edg_path.suffix.lower() == ".yml" and pred_edg_path.suffix.lower() == ".yml":
        parse_edg = parse_edge_yml
    elif (
        gt_edg_path.suffix.lower() == ".json"
        and pred_edg_path.suffix.lower() == ".json"
    ):
        parse_edg = parse_edge_json
    else:
        raise ValueError("Invalid egde-list file extension!")

    # Get the edge_dict for edge_dict_to_list()
    gt_edge_dict = parse_edg(gt_edg_path)
    pred_edge_dict = parse_edg(pred_edg_path)

    # init the graph_dict
    graph_dict = {}
    graph_dict["anterior"] = {}
    graph_dict["posterior"] = {}

    ############################################
    # First generate the graph-class (in terms of edge lists)
    # for the GT.

    gt_ant_list, gt_pos_list = edge_dict_to_list(gt_edge_dict)

    graph_dict["anterior"]["gt_graph"] = gt_ant_list
    graph_dict["posterior"]["gt_graph"] = gt_pos_list

    ############################################
    # Then for Pred

    pred_ant_list, pred_pos_list = edge_dict_to_list(pred_edge_dict)

    graph_dict["anterior"]["pred_graph"] = pred_ant_list
    graph_dict["posterior"]["pred_graph"] = pred_pos_list

    ############################################
    # The Euclidean distance between vectors
    # Add the distance to the graph_dict
    ant_dist = euclidean(gt_ant_list, pred_ant_list)
    graph_dict["anterior"]["distance"] = ant_dist

    pos_dist = euclidean(gt_pos_list, pred_pos_list)
    graph_dict["posterior"]["distance"] = pos_dist

    print("\ngraph_dict_from_files() =>")
    pprint.pprint(graph_dict, sort_dicts=False)
    return graph_dict
