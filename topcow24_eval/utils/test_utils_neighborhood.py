import numpy as np
from topcow24_eval.metrics.seg_metrics.cls_avg_b0 import connected_components
from utils_mask import filter_mask_by_label
from utils_neighborhood import get_label_neighbors


def test_get_label_neighbors_simple_np_mask():
    # Example segmentation mask array
    # 0 - background, 1, 2, 3 - labels
    mask_arr = np.array(
        [
            [0, 1, 1, 0, 2],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 3, 3],
        ]
    )
    # convert to 3D
    mask_arr = np.expand_dims(mask_arr, axis=-1)

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    # filter mask by label

    # label-1 neighbors = 3
    label = 1
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [3]

    # label-2 neighbors = 3
    label = 2
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [3]

    # label-3 neighbors = 1, 2
    label = 3
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [1, 2]


def test_get_label_neighbors_complex_np_mask():
    # Example 2D array (a segmentation mask with more labels)
    image_2d = np.array(
        [
            [0, 1, 1, 0, 2, 2, 0],
            [1, 1, 1, 0, 2, 0, 0],
            [4, 4, 0, 0, 0, 0, 3],
            [4, 4, 5, 5, 0, 3, 3],
            [0, 5, 5, 5, 6, 6, 6],
            [7, 7, 0, 6, 6, 0, 7],
        ]
    )
    # convert to 3D
    mask_arr = np.expand_dims(image_2d, axis=-1)

    # pad the original mask image
    # with background (0) around the border
    # otherwise border voxels will not have 26 neighbors for stats
    padded = np.pad(mask_arr, pad_width=1)

    # filter mask by label

    # label-1 neighbors 4
    label = 1
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [4]

    # label-2 neighbors none
    label = 2
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == []

    # label-3 neighbors 6
    label = 3
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [6]

    # label-4 neighbors 1, 5
    label = 4
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [1, 5]

    # label-5 neighbors 4, 6, 7
    label = 5
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [4, 6, 7]

    # label-6 neighbors 3, 5, 7
    label = 6
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [3, 5, 7]

    # label-7 neighbors 5, 6
    label = 7
    filtered_mask = filter_mask_by_label(padded, label)
    _, props, _ = connected_components(filtered_mask)
    neighbors = get_label_neighbors(props, padded)
    assert neighbors == [5, 6]
