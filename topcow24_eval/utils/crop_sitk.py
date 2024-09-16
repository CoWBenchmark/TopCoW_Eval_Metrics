import SimpleITK as sitk
from topcow24_eval.utils.utils_box import get_dcm_slice
from topcow24_eval.utils.utils_nii_mha_sitk import access_sitk_attr


def crop_sitk(img: sitk.Image, size_arr: list, location_arr: list) -> sitk.Image:
    # Get ROI slice
    roi_slice = get_dcm_slice(img.GetSize(), size_arr, location_arr)

    # Crop SimpleITK.Image with slice directly!
    cropped_img = img[roi_slice]

    print("\n<<< after cropping")
    print("cropped_img attr:")
    access_sitk_attr(cropped_img)

    return cropped_img
