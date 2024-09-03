"""
run the tests with pytest
"""

from pathlib import Path

from topcow24_eval.utils.utils_nii_mha_sitk import load_image_and_array_as_uint8

from .graph_classification import graph_classification

##############################################################
#   ________________________________
# < 6. Tests for graph classification >
#   --------------------------------
#          \   ^__^
#           \  (oo)\_______
#              (__)\       )\/\\
#                  ||----w |
#                  ||     ||
##############################################################

TESTDIR = Path("test_assets/seg_metrics/topcow_roi")


def test_graph_classification_topcow003():
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

    mr_img, _ = load_image_and_array_as_uint8(
        TESTDIR / f"topcow_mr_roi_{pat_id}.nii.gz"
    )
    ct_img, _ = load_image_and_array_as_uint8(
        TESTDIR / f"topcow_ct_roi_{pat_id}.nii.gz"
    )

    # both GT and Pred are MR_mask
    graph_dict = graph_classification(gt=mr_img, pred=mr_img)
    assert graph_dict == {
        "anterior": {
            "gt_graph": [0, 1, 0, 1],
            "pred_graph": [0, 1, 0, 1],
        },
        "posterior": {
            "gt_graph": [1, 1, 1, 1],
            "pred_graph": [1, 1, 1, 1],
        },
    }

    # both GT and Pred are CT_mask
    graph_dict = graph_classification(gt=ct_img, pred=ct_img)
    assert graph_dict == {
        "anterior": {
            "gt_graph": [0, 1, 0, 1],
            "pred_graph": [0, 1, 0, 1],
        },
        "posterior": {
            "gt_graph": [1, 1, 1, 1],
            "pred_graph": [1, 1, 1, 1],
        },
    }
