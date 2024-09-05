"""
Boundary intersection over union (Boundary IoU)

Metrics for Task-2-CoW-ObjDet
"""

import math

from topcow24_eval.utils.utils_box import get_end_index


def get_boundary_points_with_distances(
    size_arr: list, location_arr: list, distance_arr: list
) -> set:
    """
    Get the boundary points in coordinate tuple of (x,y,z)
    up to a distance from boundary,
    for a bounding box defined by size_arr, and location_arr

    Input:
        size_arr:
            sizes of box along the x, y, z dimension
        location_arr:
            locations of the x_min, y_min, z_min of the box
            0-indexed
        distance_arr:
            boundary distances for x, y, z
            include voxels up to that distance from boundary

    Returns:
        set of coordinates (x,y,z) that are
        within #distance from the boundary of the box
    """

    # the distances should NOT be more than half of the each size
    x_size, y_size, z_size = size_arr

    # boundary distances for x,y,z dimension
    dist_x, dist_y, dist_z = distance_arr
    assert dist_x * dist_y * dist_z > 0, "distance must be positive"

    # each distance should not be more than ceil(half of the size)
    # to prevent overshoot and save calculation
    assert dist_x <= math.ceil(x_size / 2), "distance X too big"
    assert dist_y <= math.ceil(y_size / 2), "distance Y too big"
    assert dist_z <= math.ceil(z_size / 2), "distance Z too big"

    # get min and max index of corner points of the box
    x_min, y_min, z_min = location_arr
    x_max, y_max, z_max = [
        get_end_index(loc_size_pair[0], loc_size_pair[1])
        for loc_size_pair in zip(location_arr, size_arr)
    ]

    # Generate boundary points for each face of the bounding box
    boundary_points = set()

    # There are 3 planes to add for boundary points

    # (1) Add the x-y plane on z_min and z_max
    # the x-y plane is from (x_min, y_min) to (x_max, y_max) inclusive
    # with each increment of distance in distance-Z
    # shrink the z_min and z_max by 1
    for i in range(dist_z):
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                boundary_points.add((x, y, z_min + i))
                boundary_points.add((x, y, z_max - i))

    # (2) Add the y-z plane on x_min and x_max
    # the y-z plane is from (y_min, z_min) to (y_max, z_max) inclusive
    # with each increment of distance in distance-X
    # shrink the x_min and x_max by 1
    for i in range(dist_x):
        for y in range(y_min, y_max + 1):
            for z in range(z_min, z_max + 1):
                boundary_points.add((x_min + i, y, z))
                boundary_points.add((x_max - i, y, z))

    # (3) Add the x-z plane on y_min and y_max
    # the x-z plane is from (x_min, z_min) to (x_max, z_max) inclusive
    # with each increment of distance in distance-Y
    # shrink the y_min and y_max by 1
    for i in range(dist_y):
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                boundary_points.add((x, y_min + i, z))
                boundary_points.add((x, y_max - i, z))

    # display a sorted list for debugging
    print(f"\nsorted(boundary_points):\n{sorted(boundary_points)}\n")

    return boundary_points
