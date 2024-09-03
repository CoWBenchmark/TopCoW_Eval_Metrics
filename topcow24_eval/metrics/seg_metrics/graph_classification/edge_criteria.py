"""
criteria for edges of
A1, P1

A1 := ACA_ICA_touch

P1 := PCA_BA_touch
"""

import numpy as np
from topcow24_eval.constants import MUL_CLASS_LABEL_MAP
from topcow24_eval.metrics.seg_metrics.cls_avg_b0 import connected_components
from topcow24_eval.utils.utils_mask import filter_mask_by_label, get_label_by_name
from topcow24_eval.utils.utils_neighborhood import get_label_neighbors


def has_A1(mask_arr, side="L") -> bool:
    """
    input:
        mask_arr (not padded and not filtered)
        side of "L" or "R"
    output:
        True or False

    whether the mask_arr has A1 edge
    for topcow24 public evaluation, we only adopt 1 criteria:
        {ACA-ICA touch?}
    only check if ACA and ICA are touching

    """
    # edge_name
    edge_name = f"{side}-A1"

    print(f"\n=== run has_A1(side={side}, edge_name={edge_name}) ===")

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    # get the label integers for ACA, ICA, Acom
    assert side in ["L", "R"], "unknown side"
    ACA = get_label_by_name(f"{side}-ACA", MUL_CLASS_LABEL_MAP)
    ICA = get_label_by_name(f"{side}-ICA", MUL_CLASS_LABEL_MAP)

    # filter mask by label for ACA
    ACA_mask = filter_mask_by_label(padded, ACA)

    ############################
    # check if ACA-ICA touch
    ############################

    # CC_analysis on each class separately
    print(f"\nb0cc for {side}-ACA label-{ACA}")
    _, ACA_props, _ = connected_components(ACA_mask)

    ACA_neighbors = get_label_neighbors(ACA_props, padded)

    # check if ACA and ICA touch
    if ICA in ACA_neighbors:
        print(f"{side}-ACA {ACA} and {side}-ICA {ICA} touch")
        ACA_ICA_touch = True
    else:
        print(f"{side}-ACA {ACA} and {side}-ICA {ICA} don't touch")
        ACA_ICA_touch = False

    print("\nDecision Tree")
    if not ACA_ICA_touch:
        print(f"│   ├── {side} ACA ICA dont touch")
        A1 = False
    else:
        print(f"│   ├── {side} ACA ICA touch")
        A1 = True
    print(f"\n{'└── '}{side}_A1 = {A1}\n")

    return A1


def has_P1(mask_arr, side="L") -> bool:
    """
    input:
        mask_arr (not padded and not filtered)
        side of "L" or "R"
    output:
        True or False

    whether the mask_arr has P1 edge
    for topcow24 public evaluation, we only adopt 1 criteria:
        {PCA-BA touch?}
    only check if PCA and BA are touching
    """
    # edge_name for UnusualEdgeError
    edge_name = f"{side}-P1"

    print(f"\n=== run has_P1(side={side}, edge_name={edge_name}) ===")

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    # get the label integers for PCA and BA
    assert side in ["L", "R"], "unknown side"
    PCA = get_label_by_name(f"{side}-PCA", MUL_CLASS_LABEL_MAP)
    BA = get_label_by_name("BA", MUL_CLASS_LABEL_MAP)

    # filter mask by label for PCA
    PCA_mask = filter_mask_by_label(padded, PCA)

    ############################
    # check if PCA-BA touch
    ############################

    # CC_analysis on each class separately
    print(f"\nb0cc for {side}-PCA label-{PCA}")
    _, PCA_props, _ = connected_components(PCA_mask)

    PCA_neighbors = get_label_neighbors(PCA_props, padded)

    # check if PCA and BA touch
    if BA in PCA_neighbors:
        print(f"{side}-PCA {PCA} and BA {BA} touch")
        PCA_BA_touch = True
    else:
        print(f"{side}-PCA {PCA} and BA {BA} dont touch")
        PCA_BA_touch = False

    # finally decide on the three criteria
    # traverse the decition tree from right to left
    print("\nDecision Tree")
    if not PCA_BA_touch:
        print(f"│   ├── {side} PCA BA dont touch")
        P1 = False
    else:
        print(f"│   ├── {side} PCA BA touch")
        P1 = True
    print(f"\n{'└── '}{side}_P1 = {P1}\n")

    return P1
