from pathlib import Path

from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

from .check_LR_flip import check_LR_flip

TESTDIR = Path("test_assets/seg_metrics/topcow_roi")


def test_check_LR_flip_ICA_PCA_flipped():
    """
    LPS_ICA_PCA_flipped.nii.gz has both ICAs and PCAs flipped
    the L-ICA has an outlier that is even more left than R-ICA
    but because we use median of x-index,
    the outlier does not affect the result
    """

    pred, _ = load_image_and_array_as_uint8(TESTDIR / "LPS_ICA_PCA_flipped.nii.gz")

    ant_flipped = check_LR_flip(pred, "anterior")
    pos_flipped = check_LR_flip(pred, "posterior")

    assert ant_flipped is True
    assert pos_flipped is True


def test_check_LR_flip_ACA_flipped():
    """
    LPS_ACA_flipped.nii.gz has ACAs flipped
    the L-ACA has an outlier
    but because we use median of x-index,
    the outlier does not affect the result
    """

    pred, _ = load_image_and_array_as_uint8(TESTDIR / "LPS_ACA_flipped.nii.gz")

    ant_flipped = check_LR_flip(pred, "anterior")
    pos_flipped = check_LR_flip(pred, "posterior")

    assert ant_flipped is True
    assert pos_flipped is False  # posterior is not flipped :)
