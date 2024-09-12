from pathlib import Path

import pytest
from utils_box import (
    get_dcm_slice,
    get_end_index,
    get_size,
    parse_roi_json,
    parse_roi_txt,
)

TESTDIR = Path("test_assets/box_metrics")


# big img_shape to test slice object values
BIG_IMG_SHAPE = [1e5, 1e5, 1e5]
# small img_shape to trigger out of range errors
SMALL_IMG_SHAPE = [5, 6, 7]


# fixture to reuse the example numpy arrays
@pytest.fixture
def sizes():
    return [5, 10, 1]


@pytest.fixture
def locs():
    return [0, 18, 42]


def test_parse_roi_txt():
    roi_txt = TESTDIR / "test_ROI-1_Metadata.txt"
    size_arr, location_arr = parse_roi_txt(roi_txt)

    assert size_arr == [11, 22, 33]

    assert location_arr == [0, 0, 7]


def test_parse_roi_json():
    roi_json = TESTDIR / "cow-roi.json"
    size_arr, location_arr = parse_roi_json(roi_json)

    assert size_arr == [70, 61, 17]

    assert location_arr == [35, 30, 8]


def test_get_end_index(sizes, locs):
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


def test_get_size():
    assert get_size(1, 1) == 1
    assert get_size(1, 2) == 2
    assert get_size(0, 3) == 4
    assert get_size(11, 12) == 2


def test_get_dcm_slice_fixture(sizes, locs):
    # reuse the numbers from test_get_end_index
    size_arr = sizes
    location_arr = locs

    dcm_slice = get_dcm_slice(BIG_IMG_SHAPE, size_arr, location_arr)

    assert dcm_slice == (slice(0, 5, None), slice(18, 28, None), slice(42, 43, None))


def test_get_dcm_slice_within_bound():
    size_arr, location_arr = [512, 512, 36], [0, 0, 0]
    dcm_slice = get_dcm_slice(BIG_IMG_SHAPE, size_arr, location_arr)
    assert dcm_slice == (slice(0, 512, None), slice(0, 512, None), slice(0, 36, None))

    size_arr, location_arr = [3, 4, 5], [3, 4, 5]
    dcm_slice = get_dcm_slice(BIG_IMG_SHAPE, size_arr, location_arr)
    assert dcm_slice == (slice(3, 6), slice(4, 8), slice(5, 10))

    # just nice slice for following error cases
    dcm_slice = get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=[5, 6, 7], location_arr=[0, 0, 0]
    )
    assert dcm_slice == (slice(0, 5), slice(0, 6), slice(0, 7))


def test_get_dcm_slice_crop_size_too_big():
    # any +1 in size will trigger error
    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=[5 + 1, 6, 7], location_arr=[0, 0, 0]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=[5, 6 + 1, 7], location_arr=[0, 0, 0]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=[5, 6, 7 + 1], location_arr=[0, 0, 0]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    # if return_correction=True
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=[5, 6, 7 + 1],
        location_arr=[0, 0, 0],
        return_correction=True,
    ) == ((slice(0, 5), slice(0, 6), slice(0, 7)), SMALL_IMG_SHAPE, [0, 0, 0])


def test_get_dcm_slice_neg_start():
    # negative start will trigger error
    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[-1, 0, 0]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[0, -1, 0]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[0, 0, -1]
    ) == (slice(0, 5), slice(0, 6), slice(0, 7))

    # if return_correction=True
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[-1, 0, 0],
        return_correction=True,
    ) == ((slice(0, 5), slice(0, 6), slice(0, 7)), SMALL_IMG_SHAPE, [0, 0, 0])
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[0, -1, 0],
        return_correction=True,
    ) == ((slice(0, 5), slice(0, 6), slice(0, 7)), SMALL_IMG_SHAPE, [0, 0, 0])
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[0, 0, -1],
        return_correction=True,
    ) == ((slice(0, 5), slice(0, 6), slice(0, 7)), SMALL_IMG_SHAPE, [0, 0, 0])


def test_get_dcm_slice_out_range():
    # any +1 in location will trigger error
    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[1, 0, 0]
    ) == (slice(1, 5), slice(0, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[0, 1, 0]
    ) == (slice(0, 5), slice(1, 6), slice(0, 7))

    assert get_dcm_slice(
        SMALL_IMG_SHAPE, size_arr=SMALL_IMG_SHAPE, location_arr=[0, 0, 1]
    ) == (slice(0, 5), slice(0, 6), slice(1, 7))

    # if return_correction=True
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[1, 0, 0],
        return_correction=True,
    ) == ((slice(1, 5), slice(0, 6), slice(0, 7)), [4, 6, 7], [1, 0, 0])
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[0, 1, 0],
        return_correction=True,
    ) == ((slice(0, 5), slice(1, 6), slice(0, 7)), [5, 5, 7], [0, 1, 0])
    assert get_dcm_slice(
        SMALL_IMG_SHAPE,
        size_arr=SMALL_IMG_SHAPE,
        location_arr=[0, 0, 1],
        return_correction=True,
    ) == ((slice(0, 5), slice(0, 6), slice(1, 7)), [5, 6, 6], [0, 0, 1])


def test_order_of_outOfRange():
    assert get_dcm_slice(
        img_shape=[3, 4, 5],
        size_arr=[10, 5, 5],
        location_arr=[-3, -1, 0],  # intended for enlarged box out of range
        return_correction=True,
    ) == ((slice(0, 3), slice(0, 4), slice(0, 5)), [3, 4, 5], [0, 0, 0])

    assert get_dcm_slice(
        img_shape=[3, 4, 5],
        size_arr=[10, 2, 5],
        location_arr=[-3, -1, 0],  # intended for enlarged box out of range
        return_correction=True,
    ) == ((slice(0, 3), slice(0, 2), slice(0, 5)), [3, 2, 5], [0, 0, 0])
