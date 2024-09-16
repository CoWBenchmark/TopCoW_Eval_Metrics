from os import PathLike

import SimpleITK as sitk
from topcow24_eval.utils.crop_sitk import crop_sitk
from topcow24_eval.utils.utils_box import parse_roi_txt


def crop_gt_and_pred(
    roi_txt_path: PathLike, gt: sitk.Image, pred: sitk.Image
) -> tuple[sitk.Image, sitk.Image]:
    # crop gt and pred with the same roi
    size_arr, location_arr = parse_roi_txt(roi_txt_path)

    print("gt cropped attr:")
    cropped_gt = crop_sitk(gt, size_arr, location_arr)

    print("pred cropped attr:")
    cropped_pred = crop_sitk(pred, size_arr, location_arr)

    return cropped_gt, cropped_pred
