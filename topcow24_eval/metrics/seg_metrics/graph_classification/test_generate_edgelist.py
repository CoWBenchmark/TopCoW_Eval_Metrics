from pathlib import Path

import nibabel as nib

from .generate_edgelist import generate_edgelist

TESTDIR = Path("test_assets/seg_metrics/topcow_roi")


def test_generate_edgelist_topcow003():
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
    assert generate_edgelist(mr_mask_arr) == [
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ]
    assert generate_edgelist(ct_mask_arr) == [
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ]


def test_generate_edgelist_topcow014():
    """
    topcow014

    has 3rd-A2
    014 has both P1
    A1 is there and touches ICA
    no Pcoms
    """
    pat_id = "014"

    mr_mask_arr = nib.load(TESTDIR / f"topcow_mr_roi_{pat_id}.nii.gz").get_fdata()
    ct_mask_arr = nib.load(TESTDIR / f"topcow_ct_roi_{pat_id}.nii.gz").get_fdata()

    assert generate_edgelist(mr_mask_arr) == [
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ]
    assert generate_edgelist(ct_mask_arr) == [
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ]


def test_generate_edgelist_topcow023():
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
    assert generate_edgelist(mr_mask_arr) == [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]
    assert generate_edgelist(ct_mask_arr) == [
        [0, 1, 0, 1],
        [0, 1, 0, 1],
    ]
