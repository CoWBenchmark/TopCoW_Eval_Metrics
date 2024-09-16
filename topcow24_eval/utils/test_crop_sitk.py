import numpy as np
import pytest
import SimpleITK as sitk
from crop_sitk import crop_sitk


# fixture to reuse
@pytest.fixture
def LPS_sitk_4x3x2():
    """
    [[[ 0  6 12 18]
      [ 2  8 14 20]
      [ 4 10 16 22]]

     [[ 1  7 13 19]
      [ 3  9 15 21]
      [ 5 11 17 23]]]
    """
    # Create a numpy array of shape (4, 3, 2) with unique values
    array = np.arange(24).reshape(4, 3, 2)

    # Convert the numpy array to a SimpleITK image
    # SimpleITK npy axis ordering is (z,y,x)
    image = sitk.GetImageFromArray(array.transpose((2, 1, 0)).astype(np.uint8))

    print("image.GetSize() =", image.GetSize())
    print(sitk.GetArrayViewFromImage(image))
    # sitk.WriteImage(image, "LPS_sitk_4x3x2.nii.gz", useCompression=True)

    return image


def test_crop_sitk_larger_than_img(LPS_sitk_4x3x2):
    # very large size and location wil crop to be the original image
    size_arr = [1e5, 1e5, 1e5]
    location_arr = [0, 0, 0]

    cropped = crop_sitk(
        img=LPS_sitk_4x3x2, size_arr=size_arr, location_arr=location_arr
    )

    assert np.array_equal(
        sitk.GetArrayFromImage(LPS_sitk_4x3x2), sitk.GetArrayFromImage(cropped)
    )


def test_crop_sitk_upper_right_corner_img(LPS_sitk_4x3x2):
    # crop the upper right corner
    size_arr = [2, 1, 1]
    # sitk by default is LPS+
    location_arr = [0, 2, 1]

    cropped = crop_sitk(
        img=LPS_sitk_4x3x2, size_arr=size_arr, location_arr=location_arr
    )
    # sitk.WriteImage(cropped, "test_crop_sitk_upper_right_corner_img.mha")

    assert np.array_equal(
        sitk.GetArrayFromImage(cropped),
        np.array([[[5, 11]]]),
    )


def test_crop_sitk_RAS():
    # by default sitk Images are created with LPS+
    # test with another RAS image
    mask_path = "test_assets/seg_metrics/3D/shape_5x7x9_3D_1donut_multiclass.nii.gz"
    mask = sitk.ReadImage(mask_path)

    # RAS+
    # crop a narrow rect plane
    size_arr = [3, 2, 1]
    location_arr = [1, 4, 2]

    cropped = crop_sitk(img=mask, size_arr=size_arr, location_arr=location_arr)
    # sitk.WriteImage(cropped, "test_crop_sitk_RAS.mha")
    # sitk.WriteImage(cropped, "test_crop_sitk_RAS.nii.gz", useCompression=True)

    assert np.array_equal(
        sitk.GetArrayFromImage(cropped),
        np.array(
            [
                [
                    [1, 0, 6],
                    [1, 1, 6],
                ]
            ]
        ),
    )


def test_crop_sitk_topcow():
    topcow_img = sitk.ReadImage(
        "test_assets/seg_metrics/topcow_roi/topcow_mr_roi_023.nii.gz"
    )
    cropped = crop_sitk(topcow_img, [1, 2, 1], [117, 34, 14])
    assert sitk.GetArrayFromImage(cropped).tolist() == [[[6], [0]]]

    cropped = crop_sitk(topcow_img, [2, 1, 1], [117, 34, 14])
    assert sitk.GetArrayFromImage(cropped).tolist() == [[[6, 0]]]
