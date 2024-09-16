"""
Boundary intersection over union (Boundary IoU)
from two tuples (size_arr, location_arr)

Metrics for Task-2-CoW-ObjDet
"""

import math

from .boundary_points_with_distances import boundary_points_with_distances


def iou_of_sets(first_set: set, second_set: set) -> float:
    """
    compute the intersection over union for two sets

    Inputs:
        first_set: set of coordinates (x,y,z)
        second_set: set of coordinates (x,y,z)
    Returns:
        float IoU
    """
    # Calculate intersection and union of two sets
    intersection = first_set.intersection(second_set)
    len_intersection = len(intersection)
    # print(f"len_intersection = {len_intersection}")

    union = first_set.union(second_set)
    len_union = len(union)
    # print(f"len_union = {len_union}")

    # Calculate IoU
    iou = len_intersection / len_union if len_union > 0 else 0

    return iou


def boundary_iou_from_tuple(
    first_box: tuple[list, list],
    second_box: tuple[list, list],
    boundary_distance_ratio: float,
) -> float:
    """
    calculate the boundary iou for two bounding boxes tuple

    Inputs:
        first_box:
            (tuple of the size_arr & location_arr)
        second_box:
            (tuple of the size_arr & location_arr)
        boundary_distance_ratio:
            boundary distances is a fixed ratio of each X,Y,Z size
            NOTE: for a boundary_distance_ratio of 50%,
            boundary_iou is just standard IoU

    Returns:
        boundary IoU in float
    """
    print(f"\n>>> boundary_iou_from_tuple(bdr={boundary_distance_ratio})\n")

    # use a fixed ratio for boundary distances
    size_arr_1, loc_arr_1 = first_box
    # print(f"first_box = {first_box}")
    dist_arr_1 = [math.ceil(s * boundary_distance_ratio) for s in size_arr_1]
    # print(f"dist_arr_1 = {dist_arr_1}")

    size_arr_2, loc_arr_2 = second_box
    # print(f"second_box = {second_box}")
    dist_arr_2 = [math.ceil(s * boundary_distance_ratio) for s in size_arr_2]
    # print(f"dist_arr_2 = {dist_arr_2}")

    # Get boundary points for both boxes
    # print("First box's boundary:")
    first_boundary = boundary_points_with_distances(
        size_arr=size_arr_1,
        location_arr=loc_arr_1,
        distance_arr=dist_arr_1,
    )
    # print("Second box's boundary:")
    second_boundary = boundary_points_with_distances(
        size_arr=size_arr_2,
        location_arr=loc_arr_2,
        distance_arr=dist_arr_2,
    )

    # Get the IoU of the boundary sets
    boundary_iou = iou_of_sets(first_boundary, second_boundary)

    print(f"\nboundary_iou_from_tuple() => {boundary_iou}")

    return boundary_iou
