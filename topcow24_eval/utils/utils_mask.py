from typing import List, Optional

import numpy as np


def convert_multiclass_to_binary(array: np.array) -> np.array:
    """merge all non-background labels into binary class for clDice"""
    return np.where(array > 0, True, False)


def extract_labels(array1: np.array, array2: Optional[np.array] = None) -> List[int]:
    """
    Extracts unique sorted labels from array input(s)
    if two arrays (such as gt and pred arrays) are input
    extract union of labels in gt and pred masks

    WITHOUT background 0
    """
    if array2 is not None:
        # numpy.union1d
        # Will be flattened if not already 1D
        labels = np.union1d(array1, array2)
    else:
        labels = np.unique(array1)
    print("labels = ", labels)

    # remove background 0
    filtered_labels = [int(x) for x in labels[labels != 0]]
    print(f"filtered_labels = {filtered_labels}")
    return filtered_labels


def filter_mask_by_label(mask: np.array, label: int) -> np.array:
    """
    filter the mask (numpy array), keep the voxels matching the label as 1
        convert the voxels that are not matching the label as 0
    """
    return np.where(mask == label, 1, 0)


def get_label_by_name(label_name: str, label_map: dict) -> int:
    """
    get the label intensity int value by label-name
    works with dict MUL_CLASS_LABEL_MAP
    """
    return int([k for k, v in label_map.items() if v == label_name][0])


def arr_is_binary(arr: np.array) -> bool:
    """
    test if the numpy array is binary
    NOTE: all zeros or all ones are also binary!
    """
    return set(np.unique(arr)).issubset({0, 1})
