"""
for Variant-balanced graph classification accuracy
(accuracy calculated during aggregate stage)

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from typing import Dict

import SimpleITK as sitk

from .generate_edgelist import generate_edgelist


def graph_classification(*, gt: sitk.Image, pred: sitk.Image) -> Dict:
    """
    Classify the segmentation mask according to graphs
    based on the edge lists

    Since the generation of the edge-list from a segmentation can be
    user-defined, here we generate edge-list based on the logic
    in topcow24_eval/metrics/seg_metrics/graph_classification/generate_edgelist.py.

    The reason we generate GT's edge-list and not resort to the
    Task-3 yml annotation is we want the segmentation task
    and its evaluation to be stand-alone and self-contained.
    If you are interested in outputing the edge-list as a prediction,
    and thus predict the graph class, you can turn to
    our Task-3-CoW-Classification and
    submit edge-lists/graph classes directly.

    First generate the graph-class (in terms of edge lists)
    for the GT, then generate for Pred

    Returns:
        graph_dict with
            1 = presence of edge
            0 = absence of edge
        {
            "anterior":
                {
                    "gt_graph": [1, 0, 0, 1],
                    "pred_graph": [1, 0, 1, 0],
                },
            "posterior":
                {
                    "gt_graph": [0, 0, 1, 1],
                    "pred_graph": [1, 0, 1, 1],
                },
        }
    """
    print("\nClassify CoW Graph from Seg Mask")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    graph_dict = {}
    graph_dict["anterior"] = {}
    graph_dict["posterior"] = {}

    ############################################
    # First generate the graph-class (in terms of edge lists)
    # for the GT.

    gt_mask = sitk.GetArrayFromImage(gt)

    ant_list, pos_list = generate_edgelist(gt_mask)

    graph_dict["anterior"]["gt_graph"] = ant_list
    graph_dict["posterior"]["gt_graph"] = pos_list

    ############################################
    # Then for Pred

    pred_mask = sitk.GetArrayFromImage(pred)

    ant_list, pos_list = generate_edgelist(pred_mask)

    graph_dict["anterior"]["pred_graph"] = ant_list
    graph_dict["posterior"]["pred_graph"] = pos_list

    print("\ngraph_classification() =>")
    pprint.pprint(graph_dict, sort_dicts=False)
    return graph_dict
