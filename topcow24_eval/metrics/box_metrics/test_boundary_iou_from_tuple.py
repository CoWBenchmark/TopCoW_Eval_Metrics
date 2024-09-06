import numpy as np
import SimpleITK as sitk
from topcow24_eval.metrics.seg_metrics.detection_grp2_labels import iou_single_label

from .boundary_iou_from_tuple import (
    boundary_iou_from_tuple,
    iou_of_sets,
)

##############################################################
## tests for iou_of_sets()


def test_iou_of_sets():
    # identical sets IoU = 1
    set_1 = {(3, 2, 0), (3, 2, 1), (3, 2, 2)}
    assert iou_of_sets(set_1, set_1) == 1.0

    # set_2 has one fewer item
    # IoU is 2/3
    set_2 = {(3, 2, 0), (3, 2, 1)}
    assert iou_of_sets(set_1, set_2) == 2 / 3

    # set_3 has one more item
    # IoU is 3/4
    set_3 = {(3, 2, 0), (3, 2, 1), (3, 2, 2), (2, 2, 2)}
    assert iou_of_sets(set_1, set_3) == 3 / 4

    # set_4 is empty
    # IoU is 0
    set_4 = set()
    assert iou_of_sets(set_1, set_4) == 0

    # set_5 intersection with set_1 is empty
    # IoU is 0
    set_5 = {(0, 0, 0), (111, 222, 333)}
    assert iou_of_sets(set_1, set_5) == 0

    # set_6 is union of set_5 and set_2 -> half in set_1
    # IoU = 2/5
    set_6 = set_2 | set_5
    assert iou_of_sets(set_1, set_6) == 2 / 5


##############################################################
## tests for boundary_iou_from_tuple()


def test_boundary_iou_from_tuple_tiny_vs_small():
    """
    reuse the boundary points calculated from
    - test_boundary_points_with_distances_tiny_box_surface
            size_arr = [1, 2, 3]
            location_arr = [0, 0, 0]
    - test_boundary_points_with_distances_small_box_surface
            size_arr = [3, 3, 4]
            location_arr = [0, 0, 0]

    a boundary_distance_ratio of 0.1 will have distance_arr = [1,1,1]
    so we can use the boundary points from previous test cases
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([1, 2, 3], [0, 0, 0]),
        second_box=([3, 3, 4], [0, 0, 0]),
        boundary_distance_ratio=0.1,
    )
    # all of tiny_box are inside small_box
    # so intersection = 6
    # union = small_box - 2 middle voxels = 36-2 = 34
    assert boundary_iou == 6 / 34


def test_boundary_iou_from_tuple_3D_hollow_3x3x3():
    """
    two 3x3x3 hollow cube
    first_box start at origin, second_box start at 1,1,1

    4 slices:
        slice-1: 0 intersection
        slice-2: 3 intersections
        slice-3: 3 intersections
        slice-4: 0 intersections
    = 6 intersections

    union = two 3x3x3 cubes - 1 2x2x2 cube
          = 2 * 27 - 8
          = 46

    a boundary_distance_ratio of 0.1 will have distance_arr = [1,1,1]
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([3, 3, 3], [0, 0, 0]),
        second_box=([3, 3, 3], [1, 1, 1]),
        boundary_distance_ratio=0.1,
    )
    # intersection = 6
    # union = 46
    assert boundary_iou == 6 / 46


def test_boundary_iou_from_tuple_3D_hollow_6x3x3():
    """
    two 6x3x3 hollow cube
    first_box start at origin, second_box start at 2,1,1

    4 slices:
        slice-1: 0 intersection
        slice-2: 6 intersections
        slice-3: 6 intersections
        slice-4: 0 intersections
    = 12 intersections

    union = two 6x3x3 cubes - 1 4x2x2 cube
          = 2 * 54 - 16
          = 92

    a boundary_distance_ratio of 0.2 will have distance_arr = [2,1,1]
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([6, 3, 3], [0, 0, 0]),
        second_box=([6, 3, 3], [2, 1, 1]),
        boundary_distance_ratio=0.2,
    )
    # intersection = 12
    # union = 92
    assert boundary_iou == 12 / 92 == 6 / 46


def test_boundary_iou_from_tuple_3D_hollow_6x3x3_shifted():
    """
    same as test_boundary_iou_from_tuple_3D_hollow_6x3x3
    but second_box is shifted by 1, now start at 3,1,1

    4 slices:
        slice-1: 0 intersection
        slice-2: 5 intersections
        slice-3: 5 intersections
        slice-4: 0 intersections
    = 10 intersections

    union = two 6x3x3 cubes - 1 3x2x2 cube - 2 empty
          = 2 * 54 - 12 - 2
          = 94
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([6, 3, 3], [0, 0, 0]),
        second_box=([6, 3, 3], [3, 1, 1]),
        boundary_distance_ratio=0.2,
    )
    # intersection = 10
    # union = 94
    assert boundary_iou == 10 / 94


def test_boundary_iou_from_tuple_3D_hollow_6x3x3_4x3x3():
    """
    two hollow cubes with the same hollow center
    first_box 6x3x3 start at origin, second_box 4x3x3 start at 1,0,0


    intersection = all of second_box are inside first_box
                 = 4x3x3 - 2 hollow voxels
                 = 36 - 2
                 = 34

    union = all of first box
          = 6x3x3 - 2 hollow voxels
          = 54 - 2
          = 52

    a boundary_distance_ratio of 0.2 will have
        first distance_arr = [2,1,1]
        second distance_arr = [1,1,1]
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([6, 3, 3], [0, 0, 0]),
        second_box=([4, 3, 3], [1, 0, 0]),
        boundary_distance_ratio=0.2,
    )
    # intersection = 34
    # union = 52
    assert boundary_iou == 34 / 52


def test_boundary_iou_from_tuple_3D_hollow_6x3x3_4x7x3():
    """
    two hollow cubes with concentric hollow center
    first_box 6x3x3 start at (0,2,0), second_box 4x7x3 start at (1,0,0)

    3 slices:
        slice-1: 12 intersection
        slice-2: 6 intersections
        slice-3: 12 intersections
    = 30 intersections

    since two boxes have 2 common concentric hollow voxels
    union = all of first box + 2 * 4x2x3
          = 52 + 48
          = 100

    a boundary_distance_ratio of 0.2 will have
        first distance_arr = [2,1,1]
        second distance_arr = [1,2,1]
    """
    boundary_iou = boundary_iou_from_tuple(
        first_box=([6, 3, 3], [0, 2, 0]),
        second_box=([4, 7, 3], [1, 0, 0]),
        boundary_distance_ratio=0.2,
    )
    # intersection = 30
    # union = 100
    assert boundary_iou == 30 / 100


##############################################################
## NOTE: for a boundary_distance_ratio of 50%, boundary_iou is normal IoU
# then the boundary_iou should be the same as SimpleITK overlapMeasures


def _create_sitk_image_from_bbox(size_arr, location_arr) -> sitk.Image:
    """utility to create sitk.Image from a bbox"""
    # Create a big enough numpy array with zeros
    image_np = np.zeros((20, 20, 20), dtype=np.uint8)

    # Set the voxels inside the bounding box to 1
    image_np[
        location_arr[0] : (location_arr[0] + size_arr[0]),
        location_arr[1] : (location_arr[1] + size_arr[1]),
        location_arr[2] : (location_arr[2] + size_arr[2]),
    ] = 1

    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(image_np.transpose((2, 1, 0)))

    # with random spacing
    import random

    sitk_image.SetSpacing(
        (random.randint(1, 5), random.randint(1, 5), random.randint(1, 5))
    )

    # save only for local debugging
    # sitk.WriteImage(sitk_image, f"{size_arr}{location_arr}.mha")

    return sitk_image


def test_boundary_iou_from_tuple_same_as_jaccard_345_543():
    """
    step shaped, overlap is 2x4x3
    # intersection = 3x8 = 24
    # union = 24x3 + 12x2 = 96
    """
    size_arr_1, loc_arr_1 = [3, 4, 5], [0, 0, 0]
    first_box = (size_arr_1, loc_arr_1)

    size_arr_2, loc_arr_2 = [5, 4, 3], [1, 0, 0]
    second_box = (size_arr_2, loc_arr_2)

    # create two sitk Images
    box_1_mask = _create_sitk_image_from_bbox(size_arr_1, loc_arr_1)
    box_2_mask = _create_sitk_image_from_bbox(size_arr_2, loc_arr_2)

    iou = iou_single_label(gt=box_1_mask, pred=box_2_mask, label=1)

    assert iou == 24 / 96  # 0.25

    boundary_distance_ratio = 0.5

    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )
    assert iou == boundary_iou


def test_boundary_iou_from_tuple_same_as_jaccard_tiny2x3x4():
    """
    shift horizontally the 2x3x4 solid brink
    # intersection = 3x4 = 12
    # union = 3x3x4 = 36
    """
    size_arr_1, loc_arr_1 = [2, 3, 4], [0, 0, 0]
    first_box = (size_arr_1, loc_arr_1)

    size_arr_2, loc_arr_2 = [2, 3, 4], [1, 0, 0]
    second_box = (size_arr_2, loc_arr_2)

    boundary_distance_ratio = 0.5

    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )

    assert boundary_iou == 12 / 36

    # create two sitk Images
    box_1_mask = _create_sitk_image_from_bbox(size_arr_1, loc_arr_1)
    box_2_mask = _create_sitk_image_from_bbox(size_arr_2, loc_arr_2)

    iou = iou_single_label(gt=box_1_mask, pred=box_2_mask, label=1)

    assert iou == boundary_iou


def test_boundary_iou_from_tuple_same_as_jaccard_medium_sized():
    """
    two medium-sized bbox, boundary_distance_ratio 50%
    boundary_iou == iou by sitk.jaccard
    """
    size_arr_1, loc_arr_1 = [10, 6, 9], [1, 2, 3]
    first_box = (size_arr_1, loc_arr_1)

    size_arr_2, loc_arr_2 = [9, 8, 5], [2, 1, 1]
    second_box = (size_arr_2, loc_arr_2)

    # create two sitk Images
    box_1_mask = _create_sitk_image_from_bbox(size_arr_1, loc_arr_1)
    box_2_mask = _create_sitk_image_from_bbox(size_arr_2, loc_arr_2)

    iou = iou_single_label(gt=box_1_mask, pred=box_2_mask, label=1)
    # iou_score =  0.21951219512195122

    # boundary_distance_ratio of 50%, bIOU == IOU

    boundary_distance_ratio = 0.5

    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )
    # boundary_iou_from_tuple() => 0.21951219512195122
    assert iou == boundary_iou

    # NOTE: if boundary_distance_ratio is small, this no longer holds!
    boundary_distance_ratio = 0.3
    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )
    assert iou != boundary_iou

    boundary_distance_ratio = 0.2
    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )
    assert iou != boundary_iou


def test_boundary_iou_from_tuple_same_as_jaccard_concentric():
    """
    two concentric hollow bbox, same height, middle hollow part overlap
    when boundary_distance_ratio 50%,
    boundary_iou == iou by sitk.jaccard
    """
    size_arr_1, loc_arr_1 = [9, 6, 4], [0, 0, 0]
    first_box = (size_arr_1, loc_arr_1)

    size_arr_2, loc_arr_2 = [3, 8, 4], [3, 0, 0]
    second_box = (size_arr_2, loc_arr_2)

    # create two sitk Images
    box_1_mask = _create_sitk_image_from_bbox(size_arr_1, loc_arr_1)
    box_2_mask = _create_sitk_image_from_bbox(size_arr_2, loc_arr_2)

    iou = iou_single_label(gt=box_1_mask, pred=box_2_mask, label=1)

    # iou = 3x6x4 / (9x6x4 + 3x2x4) = 72/240
    assert iou == 72 / 240 == 0.3

    # boundary_distance_ratio of 50%, bIOU == IOU

    boundary_distance_ratio = 0.5

    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )

    assert iou == boundary_iou

    # NOTE: if boundary_distance_ratio is small, this no longer holds!
    boundary_distance_ratio = 0.2
    boundary_iou = boundary_iou_from_tuple(
        first_box,
        second_box,
        boundary_distance_ratio,
    )
    # slice-1: 3x6 = 18 intersections
    # slice-2: 10 intersections
    # slice-3: 10 intersections
    # slice-4: 18 intersections
    # total = 56 intersections

    # box_1 = 9x6x4 - 5x2x2 = 196
    # box_2 = 3x8x4 - 1x4x2 = 88
    # union = 196+88-56 = 228
    assert boundary_iou == 56 / 228
    # ~= 0.24561
