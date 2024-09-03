from pathlib import Path

import nibabel as nib

from .edge_criteria import has_A1, has_P1

TESTDIR = Path("test_assets/seg_metrics/topcow_roi")


def test_edge_criteria_topcow003():
    """
    Topcow 003
        has P1
        no L-A1

        "anterior": {
            "L-A1": 0,
            "Acom": 1,
            "3rd-A2": 0,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 1,
            "L-P1": 1,
            "R-P1": 1,
            "R-Pcom": 1,
        },
    """
    pat_id = "003"

    mr_mask_arr = nib.load(TESTDIR / f"topcow_mr_roi_{pat_id}.nii.gz").get_fdata()
    ct_mask_arr = nib.load(TESTDIR / f"topcow_ct_roi_{pat_id}.nii.gz").get_fdata()

    # 003 although has a L-A2 protrusion, has no L-A1
    assert has_A1(mr_mask_arr, "L") is False
    assert has_A1(mr_mask_arr, "R")

    assert has_A1(ct_mask_arr, "L") is False
    assert has_A1(ct_mask_arr, "R")

    assert has_P1(ct_mask_arr, "L")
    assert has_P1(ct_mask_arr, "R")

    assert has_P1(mr_mask_arr, "L")
    assert has_P1(mr_mask_arr, "R")


def test_edge_criteria_topcow023():
    """
    topcow 023, no L-A1 and no R-P1

        "anterior": {
            "L-A1": 0,
            "Acom": 1,
            "3rd-A2": 0,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 0,
            "L-P1": 1,
            "R-P1": 0,
            "R-Pcom": 1,
        },
    """
    pat_id = "023"

    mr_mask_arr = nib.load(TESTDIR / f"topcow_mr_roi_{pat_id}.nii.gz").get_fdata()
    ct_mask_arr = nib.load(TESTDIR / f"topcow_ct_roi_{pat_id}.nii.gz").get_fdata()

    # topcow 023 CT same as MR
    # contains all except no L-A1 and no R-P1
    assert has_A1(mr_mask_arr, "L") is False
    assert has_A1(mr_mask_arr, "R")

    assert has_A1(ct_mask_arr, "L") is False
    assert has_A1(ct_mask_arr, "R")

    assert has_P1(ct_mask_arr, "L")
    assert has_P1(ct_mask_arr, "R") is False

    assert has_P1(mr_mask_arr, "L")
    assert has_P1(mr_mask_arr, "R") is False
