"""
for Variant-balanced topology match rate
(rate calculated during aggregate stage)

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from typing import Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from topcow24_eval.constants import (
    ANTERIOR_LABELS,
    DETECTION,
    MUL_CLASS_LABEL_MAP,
    POSTERIOR_LABELS,
)
from topcow24_eval.metrics.seg_metrics.cls_avg_b0 import connected_components
from topcow24_eval.metrics.seg_metrics.detection_grp2_labels import (
    detection_single_label,
)
from topcow24_eval.metrics.seg_metrics.graph_classification.generate_edgelist import (
    generate_edgelist,
)
from topcow24_eval.utils.utils_mask import (
    filter_mask_by_label,
)
from topcow24_eval.utils.utils_neighborhood import get_label_neighbors


def topology_matching(*, gt: sitk.Image, pred: sitk.Image) -> Dict:
    """
    #7 metric in Task-1-CoW-Segmentation.
    For labels in anterior/posterior graphs, the predicted
    segmentation needs to have
        1) Correct detection as in TP or TN
        2) Correct neighbourhood connectivity
            (connected to correct vessel classes)
        3) No 0-th Betti number errors

    This metric is a more advanced
    and stringent metric than detection (metric #5)
    and graph classification (metric #6)
    as it incorporates the detection and classification performance
    in its evaluation.

    Not trivial to get a match in our topology matching analysis ;)

    Returns
    ---
    Dict topo_dict
    e.g.
    # NOTE: b0 for A1 and P1 are actually b0 for ACA and PCA
    {
        "gt_topology": {
            "anterior": {
                "graph": [1,1,0,1],
                "L-A1": {
                    "b0": 1,
                    "neighbors": [4, 10, 12],
                },
                "Acom": {
                    "b0": 1,
                    "neighbors": [11, 12],
                },
                "3rd-A2": {
                    "b0": 0,
                    "neighbors": [],
                },
                "R-A1": {
                    "b0": 2,
                    "neighbors": [6, 10],
                },
            },
            "posterior": {
                ...
            },
        },
        "pred_topology": {
            "anterior": {
                "graph": [1,0,0,1],
                "L-A1": {
                    "detection": "FN",
                    "b0": 3,
                    "neighbors": [2, 7],
                },
                ...
            },
            "posterior": {
                ...
            },
        },
        "match_verdict": {
            "anterior": False,
            "posterior": True,
        },
    }
    """
    print("\nTopology Matching from Seg Mask")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    # init the topology dict
    topo_dict = {}

    topo_dict["gt_topology"] = {}
    topo_dict["gt_topology"]["anterior"] = {}
    topo_dict["gt_topology"]["posterior"] = {}

    gt_ant_topo = topo_dict["gt_topology"]["anterior"]
    gt_pos_topo = topo_dict["gt_topology"]["posterior"]

    topo_dict["pred_topology"] = {}
    topo_dict["pred_topology"]["anterior"] = {}
    topo_dict["pred_topology"]["posterior"] = {}

    pred_ant_topo = topo_dict["pred_topology"]["anterior"]
    pred_pos_topo = topo_dict["pred_topology"]["posterior"]

    topo_dict["match_verdict"] = {}

    ############################################
    # First gather the topology for the GT

    gt_mask = sitk.GetArrayFromImage(gt)

    # add edge-list graph entry
    ant_list, pos_list = generate_edgelist(gt_mask)

    gt_ant_topo["graph"] = ant_list
    gt_pos_topo["graph"] = pos_list

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(gt_mask, pad_width=1)

    # populate the gt topology dictionary for anterior and posteior
    populate_topo_dict(
        labels=ANTERIOR_LABELS,
        topo_dict=gt_ant_topo,
        padded=padded,
    )
    populate_topo_dict(
        labels=POSTERIOR_LABELS,
        topo_dict=gt_pos_topo,
        padded=padded,
    )

    ############################################
    # Then calculate the topology for Prediction

    pred_mask = sitk.GetArrayFromImage(pred)

    # add edge-list graph entry
    ant_list, pos_list = generate_edgelist(pred_mask)

    pred_ant_topo["graph"] = ant_list
    pred_pos_topo["graph"] = pos_list

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(pred_mask, pad_width=1)

    # populate the pred topology dictionary for anterior and posteior
    populate_topo_dict(
        labels=ANTERIOR_LABELS,
        topo_dict=pred_ant_topo,
        padded=padded,
        gt=gt,
        pred=pred,
    )
    populate_topo_dict(
        labels=POSTERIOR_LABELS,
        topo_dict=pred_pos_topo,
        padded=padded,
        gt=gt,
        pred=pred,
    )

    ############################################
    # What is the match verdict?

    # for anterior match
    print("\n--- anterior match verdict?")
    ant_match = compare_topo_dict(
        gt_ant_topo,
        pred_ant_topo,
        ANTERIOR_LABELS,
    )

    topo_dict["match_verdict"]["anterior"] = ant_match

    # for posterior match
    print("\n--- posterior match verdict?")
    pos_match = compare_topo_dict(
        gt_pos_topo,
        pred_pos_topo,
        POSTERIOR_LABELS,
    )

    topo_dict["match_verdict"]["posterior"] = pos_match

    print("\ntopology_matching() =>")
    pprint.pprint(topo_dict, sort_dicts=False)

    return topo_dict


def populate_topo_dict(
    *,
    labels: Tuple,
    topo_dict: Dict,
    padded: np.array,
    gt: Optional[sitk.Image] = None,
    pred: Optional[sitk.Image] = None,
) -> None:
    """
    populate anterior/posterior part of topology dict
        whether it is for ground truth of prediction
        depends on if gt:sitk.Image and pred:sitk.Image are supplied

    for each label in anterior or posterior labels,
        if for GT: get its b0 and neighbors
        if for Pred: get its detection, b0, neighbors

    example gt_topo_dict after processing anterior:
    {
        "Acom": {"b0": 1, "neighbors": [11, 12]},
        "R-ACA": {"b0": 1, "neighbors": [4, 10]},
        "L-ACA": {"b0": 1, "neighbors": [10]},
        "3rd-A2": {"b0": 0, "neighbors": []},
    }
    example pred_topo_dict after processing anterior:
    {
        "Acom": {"detection": "TP", "b0": 1, "neighbors": [11, 12]},
        "R-ACA": {"detection": "TP", "b0": 1, "neighbors": [4, 10]},
        "L-ACA": {"detection": "TP", "b0": 1, "neighbors": [10]},
        "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
    }
    """
    if gt is not None or pred is not None:
        for_pred = True
        print("\n*** For Prediction Topology Dict ***\n")
    else:
        for_pred = False
        print("\n*** For Ground Truth Topology Dict ***\n")

    # for each label in Anterior/Posterior labels,
    # get its (detection), b0, neighbors
    for label in labels:
        print(f"\nlabel-{label}")

        # init a field in dictionary for that label
        topo_dict[MUL_CLASS_LABEL_MAP[str(label)]] = {}
        label_topo = topo_dict[MUL_CLASS_LABEL_MAP[str(label)]]

        if for_pred:
            # detection
            detection = detection_single_label(gt=gt, pred=pred, label=label)
            label_topo["detection"] = detection

        # NOTE: need padded for N26 stats
        # filter padded mask by label
        filtered_mask = filter_mask_by_label(padded, label)
        # get the b0
        b0, label_props, _ = connected_components(filtered_mask)
        # get the neighbors
        neighbors = get_label_neighbors(label_props, padded)

        # add b0 and neighbors to topo_dict for that label
        label_topo["b0"] = int(b0)
        label_topo["neighbors"] = neighbors

    print("\nAfter mutation by populate_topo_dict():")
    pprint.pprint(topo_dict, sort_dicts=False)

    return


def compare_topo_dict(gt_topo, pred_topo, labels) -> bool:
    """
    compares gt vs pred for anterior/posterior part of the topo_dict
        topo_dicts["anterior"] OR topo_dicts["posterior"]
    labels are for anterior or posterior accordingly

    compares for
        1) if detection is only TP or TN
        2) if b0 is the same
        3) if neighbors are the same

    Returns
        True if topology is matched
    """
    print("\ncompare_topo_dict()\n")

    # see if any criteria is broken
    topo_match = True

    for label in labels:
        gt_label_topo = gt_topo[MUL_CLASS_LABEL_MAP[str(label)]]
        pred_label_topo = pred_topo[MUL_CLASS_LABEL_MAP[str(label)]]

        # 1) Correct detection as in TP or TN
        pred_detection = pred_label_topo["detection"]

        # print("pred_detection = ", pred_detection)

        if pred_detection not in (DETECTION.TP.value, DETECTION.TN.value):
            print("[X] detection not matched")
            topo_match = False
            break

        # 2) Correct neighbourhood connectivity
        #     (connected to correct vessel classes)
        gt_neighbors = gt_label_topo["neighbors"]
        pred_neighbors = pred_label_topo["neighbors"]

        if gt_neighbors != pred_neighbors:
            print("[X] neighbors not matched")
            topo_match = False
            break

        # 3) No 0-th Betti number errors
        gt_b0 = gt_label_topo["b0"]
        pred_b0 = pred_label_topo["b0"]

        if gt_b0 != pred_b0:
            print("[X] b0 not matched")
            topo_match = False
            break

    # only then is topology matched :)

    print("topo_match = ", topo_match)
    return topo_match
