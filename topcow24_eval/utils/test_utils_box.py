from pathlib import Path

from utils_box import parse_roi_txt

TESTDIR = Path("test_assets/box_metrics")


def test_parse_roi_txt():
    roi_txt = TESTDIR / "test_ROI-1_Metadata.txt"
    size_arr, location_arr = parse_roi_txt(roi_txt)

    assert size_arr == [11, 22, 33]

    assert location_arr == [0, 0, 7]
