import numpy as np
import pytest
import SimpleITK as sitk
from topcow24_eval.constants import MUL_CLASS_LABEL_MAP
from utils_mask import (
    arr_is_binary,
    convert_multiclass_to_binary,
    extract_labels,
    filter_mask_by_label,
    get_label_by_name,
    pad_sitk_image,
)


def test_convert_multiclass_to_binary():
    # np.arange will be converted to all True except first item
    mul_mask = np.arange(6).reshape(3, 2)
    np.testing.assert_equal(
        convert_multiclass_to_binary(mul_mask),
        np.array([[0, 1], [1, 1], [1, 1]]),
    )

    mul_mask = np.arange(27).reshape(3, 3, 3)
    fst_zero = np.ones((3, 3, 3))
    fst_zero[0][0][0] = 0
    np.testing.assert_equal(convert_multiclass_to_binary(mul_mask), fst_zero)

    # all zeroes will stay all zeroes
    np.testing.assert_equal(
        convert_multiclass_to_binary(np.zeros((3, 2, 4))), np.zeros((3, 2, 4))
    )

    # all ones will stay all ones
    np.testing.assert_equal(
        convert_multiclass_to_binary(np.ones((3, 2, 4))), np.ones((3, 2, 4))
    )


def test_extract_labels():
    assert extract_labels(
        array1=np.array([1, 2, 3]),
        array2=np.array([3, 2, 1]),
    ) == [1, 2, 3]

    # will remove background
    assert extract_labels(
        array1=np.array([0, 1, 2, 3]),
    ) == [1, 2, 3]

    # will de-dup the labels
    assert extract_labels(
        array1=np.array([0, 13, 5, 1, 13]),
        array2=np.array([5, 5, 5]),
    ) == [1, 5, 13]

    # wil be sorted
    assert extract_labels(
        array1=np.array([0, 5, 4, 3]),
        array2=np.array([3, 0]),
    ) == [3, 4, 5]


def test_filter_mask_by_label():
    # filter a 1D arr by label=2
    np.testing.assert_equal(
        filter_mask_by_label(np.array([0, 1, 2, 3, 4, 5]), label=2),
        np.array([0, 0, 1, 0, 0, 0]),
    )

    # filter a 2D arr
    arr2D = np.array([[1, 2], [3, 4], [5, 6]])
    # by label=7 will be all zero
    np.testing.assert_array_equal(
        filter_mask_by_label(arr2D, label=7), np.zeros((3, 2))
    )
    # BUT by label=6 will be have one pixel as foreground
    np.testing.assert_array_equal(
        filter_mask_by_label(arr2D, label=6), np.array([[0, 0], [0, 0], [0, 1]])
    )


def test_get_label_by_name():
    # works for the multiclass label_map
    assert get_label_by_name("3rd-A2", MUL_CLASS_LABEL_MAP) == 15
    assert get_label_by_name("BA", MUL_CLASS_LABEL_MAP) == 1
    assert get_label_by_name("Background", MUL_CLASS_LABEL_MAP) == 0
    assert get_label_by_name("L-ACA", MUL_CLASS_LABEL_MAP) == 12

    # unrecognized label_name will cause error
    with pytest.raises(IndexError) as e_info:
        get_label_by_name("KON", MUL_CLASS_LABEL_MAP)
    assert "list index out of range" in str(e_info)


def test_arr_is_binary():
    # all zero or all ones also works!
    assert arr_is_binary(np.zeros((3, 2))) is True
    assert arr_is_binary(np.ones((3, 2))) is True

    # mixed 01
    assert arr_is_binary(np.array([0, 1, 0, 1])) is True

    # bool
    assert arr_is_binary(np.eye(4, dtype=bool)) is True

    # multiclass
    assert arr_is_binary(np.arange(6).reshape(3, 2)) is False
    assert arr_is_binary(np.arange(6).reshape(3, 2).astype(bool)) is True


######
# pad SimpleITK image
def test_pad_sitk_image():
    # a 2D example
    img_2D = sitk.Image([2, 3], sitk.sitkUInt8)
    img_2D = sitk.Add(img_2D, 1)  # Fill with 1s

    padded_2D = pad_sitk_image(img_2D)

    assert np.array_equal(
        sitk.GetArrayFromImage(padded_2D),
        np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        ),
    )
    print("original: ", sitk.GetArrayFromImage(img_2D))

    # a 3D example
    image1 = sitk.Image([2, 2, 2], sitk.sitkUInt8)
    # Fill with 42s
    image1[:, :, :] = 42

    padded = pad_sitk_image(image1)

    assert np.array_equal(
        sitk.GetArrayFromImage(padded),
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 42, 42, 0],
                    [0, 42, 42, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 42, 42, 0],
                    [0, 42, 42, 0],
                    [0, 0, 0, 0],
                ],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            ]
        ),
    )
    print("original: ", sitk.GetArrayFromImage(image1))
