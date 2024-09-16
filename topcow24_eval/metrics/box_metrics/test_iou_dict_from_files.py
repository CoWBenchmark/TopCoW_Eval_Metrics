"""
run the tests with pytest
"""

from pathlib import Path

from .iou_dict_from_files import iou_dict_from_files

##############################################################
#   ________________________________
# < 8. Tests for boundary IoU >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR = Path("test_assets/box_metrics")


def test_iou_dict_from_files_tiny_vs_small():
    """
    reuse test_boundary_iou_from_tuple_tiny_vs_small()
    - test_boundary_points_with_distances_tiny_box_surface
            size_arr = [1, 2, 3]
            location_arr = [0, 0, 0]
    - test_boundary_points_with_distances_small_box_surface
            size_arr = [3, 3, 4]
            location_arr = [0, 0, 0]

    Boundary IoU at boundary_distance_ratio of 0.2 will still
    have distance_arr = [1,1,1]
    same as in test_boundary_iou_from_tuple_tiny_vs_small()
    # all of tiny_box are inside small_box
    # so intersection = 6
    # union = small_box - 2 middle voxels = 36-2 = 34
    assert boundary_iou == 6 / 34

    standard IoU:
        intersection = 1x2x3 = 6
        union = 3x3x4 = 36
    """
    iou_dict = iou_dict_from_files(
        TESTDIR / "test_iou_dict_from_files_tiny_vs_small_box_1.json",
        TESTDIR / "test_iou_dict_from_files_tiny_vs_small_box_2.txt",
    )
    assert iou_dict == {
        "Boundary_IoU": 6 / 34,
        "IoU": 6 / 36,
    }


def test_iou_dict_from_files_hollow_6x3x3_shifted():
    """
    reuse test_boundary_iou_from_tuple_3D_hollow_6x3x3_shifted()

    # intersection = 10
    # union = 94
    assert boundary_iou == 10 / 94

    IoU = 1 3x2x2 cube / (two 6x3x3 cubes - 1 3x2x2 cube)
        = 12/96
    """
    iou_dict = iou_dict_from_files(
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_shifted_box_1.json",
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_shifted_box_2.txt",
    )
    assert iou_dict == {
        "Boundary_IoU": 10 / 94,
        "IoU": 12 / 96,
    }


def test_iou_dict_from_files_hollow_6x3x3_4x3x3():
    """
    reuse test_boundary_iou_from_tuple_3D_hollow_6x3x3_4x3x3()

    # intersection = 34
    # union = 52
    assert boundary_iou == 34 / 52

    Since all of second_box are inside first_box
    IoU = 36 / 54
    """
    iou_dict = iou_dict_from_files(
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_4x3x3_box_1.JSON",
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_4x3x3_box_2.TXT",
    )
    assert iou_dict == {
        "Boundary_IoU": 34 / 52,
        "IoU": 36 / 54,
    }


def test_iou_dict_from_files_hollow_6x3x3_4x7x3():
    """
    reuse test_boundary_iou_from_tuple_3D_hollow_6x3x3_4x7x3()

    # intersection = 30
    # union = 100
    assert boundary_iou == 30 / 100

    IoU = 4x3x3 / 34x3
        = 36 / 102
    """
    iou_dict = iou_dict_from_files(
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_4x7x3_box_1.TXT",
        TESTDIR / "test_iou_dict_from_files_hollow_6x3x3_4x7x3_box_2.JSON",
    )
    assert iou_dict == {
        "Boundary_IoU": 30 / 100,
        "IoU": 36 / 102,
    }
