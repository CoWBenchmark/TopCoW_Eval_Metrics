from pathlib import Path

from utils_box import get_end_index, parse_roi_txt

TESTDIR = Path("test_assets/box_metrics")


def test_parse_roi_txt():
    roi_txt = TESTDIR / "test_ROI-1_Metadata.txt"
    size_arr, location_arr = parse_roi_txt(roi_txt)

    assert size_arr == [11, 22, 33]

    assert location_arr == [0, 0, 7]


def test_get_end_index():
    sizes = [5, 10, 1]
    locs = [0, 18, 42]

    start_index = locs[0]
    size = sizes[0]
    end_index = get_end_index(start_index, size)
    assert end_index == 4

    start_index = locs[1]
    size = sizes[1]
    end_index = get_end_index(start_index, size)
    assert end_index == 27

    start_index = locs[2]
    size = sizes[2]
    end_index = get_end_index(start_index, size)
    assert end_index == 42
