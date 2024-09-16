"""
utils for neighborhood (N26) analysis
- get indices/coords of 26 neighborhoods of a point or a blob
- get values of the N26 neighbors of a point or a blob
- get statistics of the 26 neighbors
...

NOTE: N26_utils assumes the img to be padded,
        otherwise border voxels will not have 26 neighbors for stats
"""

import numpy as np
from topcow24_eval.utils.utils_mask import extract_labels


def get_N26_ind(p):
    """
    get indices of 26 neighborhoods of a point p in 3D image
        i.e. connectivity of 3
    ref: https://stackoverflow.com/questions/71134868/get-26-nearest-neighbors-of-a-point-in-3d-space-vectorized

    input:
        p = [x, y, z] coord of a point
    output:
        array with shape (26, 3) of the coords of the 26 neighbours
    """
    motion = np.transpose(np.indices((3, 3, 3)) - 1).reshape(-1, 3)
    # remove origin (0, 0, 0) in the middle with index=13
    motion = np.delete(motion, 13, axis=0)

    # expand dims for point p
    p = np.expand_dims(p, axis=0)
    N26_ind = motion + p
    return N26_ind


def get_coords_N26_ind(coords):
    """
    get_N26_ind() but for a list of coords from region.coords (N, 3) ndarray

    Firstly gather all the N26 indices for all voxels in coords
    Secondly de-duplicate the indices
    Thirdly remove the indices already in coords

    input:
        the region.coords (N, 3) ndarray
            e.g. 3D -> region.coords:
                    [[38 82  8]
                    [38 83  8]
                    [39 82  6]
                    [39 82  7]
                    [39 82  8]
                    [39 83  8]] (6, 3)

    output:
        sorted array with shape (X, 3) of all the neighbouring voxels of coords
    """
    ####################################################################
    # Firstly gather all the N26 indices for all voxels in coords
    ####################################################################

    # reuse the first coord to build the all indices array
    all_N26_ind = np.expand_dims(coords[0], axis=0)

    for p in coords:
        all_N26_ind = np.concatenate((get_N26_ind(p), all_N26_ind))

    # print("all_N26_ind =\n", all_N26_ind)

    ####################################################################
    # Secondly de-duplicate the indices
    ####################################################################

    # sorted unique elements of an array
    unique_ind = np.unique(all_N26_ind, axis=0)

    # print("unique_ind =\n", unique_ind)

    ####################################################################
    # Thirdly remove the indices already in coords
    ####################################################################
    neighbors_only = mazdak_broadcasting_approach(unique_ind, coords)

    # print("neighbors_only=\n", neighbors_only)

    return neighbors_only


def mazdak_broadcasting_approach(A, B):
    """
    get the complementary set of A-B, remove elements from A that are in B

    with broadcasting

    ref https://stackoverflow.com/questions/40055835/removing-elements-from-an-array-that-are-in-another-array
    """
    return A[np.all(np.any((A - B[:, None]), axis=2), axis=0)]


def get_coords_N26_val(coords, img):
    """
    get_N26_val() but for list of coords as input

    input:
        the region.coords (N, 3) ndarray
        and the corresponding img array
    output:
        array with shape (X,) of values of all the neighbours of coords
    """
    indices = get_coords_N26_ind(coords)

    return img[
        indices[:, 0],
        indices[:, 1],
        indices[:, 2],
    ]


def get_N26_stats(N26_vals):
    """
    get statistics of the 26 neighbors
    based on the values array from get_N26_val() or get_coords_N26_val()
    ref: https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-in-an-ndarray

    input:
        array with shape (X,) of values of all neighbours
    output:
        dict of {val: count} from np.unique()
    """
    unique, counts = np.unique(N26_vals, return_counts=True)

    return dict(zip(unique.astype(int), counts.astype(int)))


def get_region_N26_stats(region, padded):
    """
    get the N26 stats for a region from regionprops
    padded is the padded original mask, and unfiltered
    """
    # get the N26 values using get_coords_N26_val()
    vals = get_coords_N26_val(region.coords, padded)

    # get the N26 stats based on the values
    N26_stats = get_N26_stats(vals)

    # print(f"before pop0, N26_stats = {N26_stats}")

    return N26_stats


def get_label_neighbors(label_props, padded) -> list[int]:
    """
    get the neighboring labels for a label's regionprops

    input:
        label_props from measure.regionprops() for a label
        padded unfiltered mask array
    output:
        sorted List from np.unique of neighbors
        without background 0
    """
    neighbors = []

    for region in label_props:
        # print("\nregion.area = ", region.area)
        # # Set numpy to summarize >20 size array for neatness
        # with np.printoptions(suppress=True, threshold=20):
        #     print("region.coords:\n", region.coords, region.coords.shape)

        # NOTE: get_region_N26_stats() works with padded
        N26_stats = get_region_N26_stats(region, padded)

        # combine N26_stats
        neighbors += N26_stats

    neighbors = extract_labels(neighbors)

    print("neighbors = ", neighbors)

    return neighbors
