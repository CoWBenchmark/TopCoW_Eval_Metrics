"""
Calculate the Dict of
Boundary intersection over union (Boundary IoU)
and IoU
from two roi txt|json files

Metrics for Task-2-CoW-ObjDet
"""

import os
from pathlib import Path

from topcow24_eval.constants import BOUNDARY_DISTANCE_RATIO, MAX_DISTANCE_RATIO
from topcow24_eval.utils.utils_box import parse_roi_json, parse_roi_txt

from .boundary_iou_from_tuple import boundary_iou_from_tuple


def iou_dict_from_files(
    first_box_path: str | os.PathLike, second_box_path: str | os.PathLike
) -> dict:
    """
    wrap the boundary_iou_from_tuple() with file Path as inputs
    read the file (.txt or .json) from paths
    extrat the tuple of the (size_arr, location_arr)
    and call boundary_iou_from_tuple() twice:
        once with boundary_distance_ratio set by constants.py
        then again for a standard IoU
    return a dict {"IoU" and "Boundary IoU"}

    NOTE: bbox roi files should be .txt or .json
    example json:
        {"size": [70, 61, 17], "location": [35, 30, 8]}
    example txt:
        --- ROI Meta Data ---
        Size (Voxels): 96 68 36
        Location (Voxels): 103 93 77

    Inputs:
        first_box_path:
            path of the first bounding box, {.json|.txt}
        second_box_path:
            path of the second bounding box, {.json|.txt}

    Returns:
        dict of iou and boundary_iou
    """
    print("\n>>> iou_dict_from_files >>>\n")

    first_box_path = Path(first_box_path)
    second_box_path = Path(second_box_path)

    assert first_box_path.is_file(), "first_box_path don't exist!"
    assert second_box_path.is_file(), "second_box_path don't exist!"

    if (
        first_box_path.suffix.lower() == ".txt"
        and second_box_path.suffix.lower() == ".txt"
    ):
        parse_roi = parse_roi_txt
    elif (
        first_box_path.suffix.lower() == ".json"
        and second_box_path.suffix.lower() == ".json"
    ):
        parse_roi = parse_roi_json
    else:
        raise ValueError("Invalid roi file extension!")

    first_box = parse_roi(first_box_path)
    second_box = parse_roi(second_box_path)

    iou_dict = {}

    # call boundary_iou_from_tuple() twice:

    # once with boundary_distance_ratio set by constants.py
    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        BOUNDARY_DISTANCE_RATIO,
    )
    # then again for a standard IoU
    iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        MAX_DISTANCE_RATIO,
    )

    iou_dict["Boundary IoU"] = boundary_iou
    iou_dict["IoU"] = iou

    print(f"\niou_dict_from_files() => {iou_dict}")

    return iou_dict
