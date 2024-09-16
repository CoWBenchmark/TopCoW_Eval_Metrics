"""
NOTE: important to turn off the mirror augmentation in nnUnet
for multiclass segmentation to avoid left and right labels
being wrongly flipped such as in some predictions
"""

import numpy as np
import SimpleITK as sitk
from topcow24_eval.constants import MUL_CLASS_LABEL_MAP
from topcow24_eval.utils.utils_mask import (
    extract_labels,
    get_label_by_name,
)


def check_LR_flip(pred: sitk.Image, region: str) -> bool:
    """
    Check if the mask_arr mirror-flipped the left and right labels

    The topcow data are all in LPS+, so L and R are along the x-axis
    with Right-most at x-0 ad Left-most at x-max.
    Get the medians of x-index of the binarized labels,
    and since the image is LPS+, left organs should have
    bigger x values than right organs.

    Check for either anterior or posterior with region string

    Params
        pred:
            sitk.Image of the mask
        region:
            "anterior" or "posterior"

    Returns
        whether left-right organs flipped
    """
    assert region in ("anterior", "posterior"), "invalid region"

    print(f"\ncheck_LR_flip(region={region})\n")

    # NOTE: SimpleITK npy axis ordering is (z,y,x)!
    # reorder from (z,y,x) to (x,y,z)
    mask_arr = sitk.GetArrayFromImage(pred).transpose((2, 1, 0)).astype(np.uint8)

    # get the unique labels
    labels = extract_labels(mask_arr)

    if region == "anterior":
        # for anterior, if ICAs exist, use ICAs to check LR flip
        # if ICAs do not exist, resort to ACAs

        # get the label integers for ICAs
        label_L_ICA = get_label_by_name("L-ICA", MUL_CLASS_LABEL_MAP)
        label_R_ICA = get_label_by_name("R-ICA", MUL_CLASS_LABEL_MAP)

        # get the label integers for ACAs
        label_L_ACA = get_label_by_name("L-ACA", MUL_CLASS_LABEL_MAP)
        label_R_ACA = get_label_by_name("R-ACA", MUL_CLASS_LABEL_MAP)

        if {label_L_ICA, label_R_ICA}.issubset(set(labels)):
            # print("Use ICA")
            left_organ = np.median(np.where(mask_arr == label_L_ICA)[0])
            right_organ = np.median(np.where(mask_arr == label_R_ICA)[0])
        elif {label_L_ACA, label_R_ACA}.issubset(set(labels)):
            # print("Use ACA")
            left_organ = np.median(np.where(mask_arr == label_L_ACA)[0])
            right_organ = np.median(np.where(mask_arr == label_R_ACA)[0])
        else:
            print("ICAs, ACAs not in labels, cannot check anterior LR flip")
            return False

    else:
        # for posterior, use PCAs to check LR flip

        # get the label integers for PCAs
        label_L_PCA = get_label_by_name("L-PCA", MUL_CLASS_LABEL_MAP)
        label_R_PCA = get_label_by_name("R-PCA", MUL_CLASS_LABEL_MAP)

        if {label_L_PCA, label_R_PCA}.issubset(set(labels)):
            # print("Use PCA")
            left_organ = np.median(np.where(mask_arr == label_L_PCA)[0])
            right_organ = np.median(np.where(mask_arr == label_R_PCA)[0])
        else:
            print("PCAs not in labels, cannot check posterior LR flip")
            return False

    # check for LR flip using the median x index

    # print(f"left_organ = {left_organ}")
    # print(f"right_organ = {right_organ}")

    flipped = bool(left_organ < right_organ)
    print(f"flipped? {flipped}")

    return flipped
