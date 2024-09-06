import pytest

from .boundary_points_with_distances import boundary_points_with_distances

##############################################################
## tests for boundary_points_with_distances()


def test_boundary_points_with_distances_tiny_box_surface():
    """
    tiny bounding box with size_arr = [1, 2, 3]
    starting at origin,

    get the boundary points on the surface with distances [1,1,1]
    """
    size_arr = [1, 2, 3]
    location_arr = [0, 0, 0]
    distance_arr = [1, 1, 1]
    boundary_points = boundary_points_with_distances(
        size_arr, location_arr, distance_arr
    )
    # all 6 points of the box are included
    assert boundary_points == {
        (0, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
        (0, 0, 2),
        (0, 1, 2),
    }

    # NOTE: adding distances to this tiny box
    # should raise distance too big error

    with pytest.raises(AssertionError) as e_info:
        boundary_points_with_distances(size_arr, location_arr, [2, 1, 1])
    assert str(e_info.value) == "distance X too big"

    with pytest.raises(AssertionError) as e_info:
        boundary_points_with_distances(size_arr, location_arr, [1, 2, 1])
    assert str(e_info.value) == "distance Y too big"

    # Size-Z is 3, so distance-Z of 2 is still okay!
    assert boundary_points == boundary_points_with_distances(
        size_arr,
        location_arr,
        [1, 1, 2],
    )

    # but distance-Z of 3 will trigger error!
    with pytest.raises(AssertionError) as e_info:
        boundary_points_with_distances(size_arr, location_arr, [1, 1, 3])
    assert str(e_info.value) == "distance Z too big"


def test_boundary_points_with_distances_small_box_surface():
    """
    small (bigger than tiny) bounding box with size_arr = [3, 3, 4]
    starting at origin,

    get the boundary points on the surface with distances [1,1,1]
    """
    size_arr = [3, 3, 4]
    location_arr = [0, 0, 0]
    distance_arr = [1, 1, 1]
    boundary_points = boundary_points_with_distances(
        size_arr, location_arr, distance_arr
    )
    # all points of the box, except the center 2, are included
    assert boundary_points == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 0),
        (0, 2, 1),
        (0, 2, 2),
        (0, 2, 3),
        (1, 0, 0),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
        (1, 1, 0),
        # (1, 1, 1),  # middle 2 voxels
        # (1, 1, 2),  # middle 2 voxels
        (1, 1, 3),
        (1, 2, 0),
        (1, 2, 1),
        (1, 2, 2),
        (1, 2, 3),
        (2, 0, 0),
        (2, 0, 1),
        (2, 0, 2),
        (2, 0, 3),
        (2, 1, 0),
        (2, 1, 1),
        (2, 1, 2),
        (2, 1, 3),
        (2, 2, 0),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
    }


def test_boundary_points_with_distances_small_box_D111():
    """
    small (bigger than tiny) bounding box with size_arr = [5, 5, 3]
    starting at (3,2,0),

    get the boundary points on the surface with
        distances [1,1,1]
    """
    size_arr = [5, 5, 3]
    location_arr = [3, 2, 0]
    distance_arr = [1, 1, 1]
    boundary_points = boundary_points_with_distances(
        size_arr, location_arr, distance_arr
    )
    # all points of the box, except the middle 9, are included
    assert boundary_points == {
        (3, 2, 0),
        (3, 2, 1),
        (3, 2, 2),
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 2),
        (3, 5, 0),
        (3, 5, 1),
        (3, 5, 2),
        (3, 6, 0),
        (3, 6, 1),
        (3, 6, 2),
        (4, 2, 0),
        (4, 2, 1),
        (4, 2, 2),
        (4, 3, 0),
        # (4, 3, 1),
        (4, 3, 2),
        (4, 4, 0),
        # (4, 4, 1),
        (4, 4, 2),
        (4, 5, 0),
        # (4, 5, 1),
        (4, 5, 2),
        (4, 6, 0),
        (4, 6, 1),
        (4, 6, 2),
        (5, 2, 0),
        (5, 2, 1),
        (5, 2, 2),
        (5, 3, 0),
        # (5, 3, 1),
        (5, 3, 2),
        (5, 4, 0),
        # (5, 4, 1),
        (5, 4, 2),
        (5, 5, 0),
        # (5, 5, 1),
        (5, 5, 2),
        (5, 6, 0),
        (5, 6, 1),
        (5, 6, 2),
        (6, 2, 0),
        (6, 2, 1),
        (6, 2, 2),
        (6, 3, 0),
        # (6, 3, 1),
        (6, 3, 2),
        (6, 4, 0),
        # (6, 4, 1),
        (6, 4, 2),
        (6, 5, 0),
        # (6, 5, 1),
        (6, 5, 2),
        (6, 6, 0),
        (6, 6, 1),
        (6, 6, 2),
        (7, 2, 0),
        (7, 2, 1),
        (7, 2, 2),
        (7, 3, 0),
        (7, 3, 1),
        (7, 3, 2),
        (7, 4, 0),
        (7, 4, 1),
        (7, 4, 2),
        (7, 5, 0),
        (7, 5, 1),
        (7, 5, 2),
        (7, 6, 0),
        (7, 6, 1),
        (7, 6, 2),
    }


def test_boundary_points_with_distances_small_box_D121():
    """
    same as test_boundary_points_with_distances_small_box_D111()
    but now:
        distances [1,2,1]
    d_y squeeze into the original 9 voxel middle plane,
    now left with just a 3-voxel row
    """
    size_arr = [5, 5, 3]
    location_arr = [3, 2, 0]
    distance_arr = [1, 2, 1]
    boundary_points = boundary_points_with_distances(
        size_arr, location_arr, distance_arr
    )
    # all points of the box, except the middle 3
    # shrunk due to d_y, are included
    assert boundary_points == {
        (3, 2, 0),
        (3, 2, 1),
        (3, 2, 2),
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 2),
        (3, 5, 0),
        (3, 5, 1),
        (3, 5, 2),
        (3, 6, 0),
        (3, 6, 1),
        (3, 6, 2),
        (4, 2, 0),
        (4, 2, 1),
        (4, 2, 2),
        (4, 3, 0),
        (4, 3, 1),
        (4, 3, 2),
        (4, 4, 0),
        # (4, 4, 1),  # squeezed by d_y + 1
        (4, 4, 2),
        (4, 5, 0),
        (4, 5, 1),
        (4, 5, 2),
        (4, 6, 0),
        (4, 6, 1),
        (4, 6, 2),
        (5, 2, 0),
        (5, 2, 1),
        (5, 2, 2),
        (5, 3, 0),
        (5, 3, 1),
        (5, 3, 2),
        (5, 4, 0),
        # (5, 4, 1),  # squeezed by d_y + 1
        (5, 4, 2),
        (5, 5, 0),
        (5, 5, 1),
        (5, 5, 2),
        (5, 6, 0),
        (5, 6, 1),
        (5, 6, 2),
        (6, 2, 0),
        (6, 2, 1),
        (6, 2, 2),
        (6, 3, 0),
        (6, 3, 1),
        (6, 3, 2),
        (6, 4, 0),
        # (6, 4, 1),  # squeezed by d_y + 1
        (6, 4, 2),
        (6, 5, 0),
        (6, 5, 1),
        (6, 5, 2),
        (6, 6, 0),
        (6, 6, 1),
        (6, 6, 2),
        (7, 2, 0),
        (7, 2, 1),
        (7, 2, 2),
        (7, 3, 0),
        (7, 3, 1),
        (7, 3, 2),
        (7, 4, 0),
        (7, 4, 1),
        (7, 4, 2),
        (7, 5, 0),
        (7, 5, 1),
        (7, 5, 2),
        (7, 6, 0),
        (7, 6, 1),
        (7, 6, 2),
    }


def test_boundary_points_with_distances_small_box_D211():
    """
    same as test_boundary_points_with_distances_small_box_D121()
    but now:
        distances [2,1,1]
    d_x squeeze into the original 9 voxel middle plane,
    now left with just a 3-voxel column
    """
    size_arr = [5, 5, 3]
    location_arr = [3, 2, 0]
    distance_arr = [2, 1, 1]
    boundary_points = boundary_points_with_distances(
        size_arr, location_arr, distance_arr
    )
    # all points of the box, except the middle 3
    # shrunk due to d_x, are included
    assert boundary_points == {
        (3, 2, 0),
        (3, 2, 1),
        (3, 2, 2),
        (3, 3, 0),
        (3, 3, 1),
        (3, 3, 2),
        (3, 4, 0),
        (3, 4, 1),
        (3, 4, 2),
        (3, 5, 0),
        (3, 5, 1),
        (3, 5, 2),
        (3, 6, 0),
        (3, 6, 1),
        (3, 6, 2),
        (4, 2, 0),
        (4, 2, 1),
        (4, 2, 2),
        (4, 3, 0),
        (4, 3, 1),
        (4, 3, 2),
        (4, 4, 0),
        (4, 4, 1),
        (4, 4, 2),
        (4, 5, 0),
        (4, 5, 1),
        (4, 5, 2),
        (4, 6, 0),
        (4, 6, 1),
        (4, 6, 2),
        (5, 2, 0),
        (5, 2, 1),
        (5, 2, 2),
        (5, 3, 0),
        # (5, 3, 1),  # squeezed by d_x + 1
        (5, 3, 2),
        (5, 4, 0),
        # (5, 4, 1),  # squeezed by d_x + 1
        (5, 4, 2),
        (5, 5, 0),
        # (5, 5, 1),  # squeezed by d_x + 1
        (5, 5, 2),
        (5, 6, 0),
        (5, 6, 1),
        (5, 6, 2),
        (6, 2, 0),
        (6, 2, 1),
        (6, 2, 2),
        (6, 3, 0),
        (6, 3, 1),
        (6, 3, 2),
        (6, 4, 0),
        (6, 4, 1),
        (6, 4, 2),
        (6, 5, 0),
        (6, 5, 1),
        (6, 5, 2),
        (6, 6, 0),
        (6, 6, 1),
        (6, 6, 2),
        (7, 2, 0),
        (7, 2, 1),
        (7, 2, 2),
        (7, 3, 0),
        (7, 3, 1),
        (7, 3, 2),
        (7, 4, 0),
        (7, 4, 1),
        (7, 4, 2),
        (7, 5, 0),
        (7, 5, 1),
        (7, 5, 2),
        (7, 6, 0),
        (7, 6, 1),
        (7, 6, 2),
    }
