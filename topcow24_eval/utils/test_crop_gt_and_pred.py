from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from crop_gt_and_pred import crop_gt_and_pred


def test_crop_gt_and_pred():
    """reuse test_crop_sitk_RAS()"""

    TESTDIR_3D = Path("test_assets/seg_metrics/3D")

    # expected crops:
    # cropped shape_5x7x9_3D_1donut.nii.gz
    # save as RAS affine
    # NOTE: nifti np array is (x,y,z), SimpleITK npy axis ordering is (z,y,x)
    expected_1donut_gt_data = np.array(
        [[[1], [1]], [[0], [1]], [[1], [1]]], dtype=np.uint8
    )
    expected_gt_path = TESTDIR_3D / "expected_1donut_gt_crop.nii.gz"
    nib.Nifti1Image(expected_1donut_gt_data, np.eye(4)).to_filename(
        expected_gt_path,
    )

    # cropped shape_5x7x9_3D_1donut_multiclass.nii.gz
    expected_1donut_pred_data = np.array(
        [[[1], [1]], [[0], [1]], [[6], [6]]], dtype=np.uint8
    )
    expected_pred_path = TESTDIR_3D / "expected_1donut_pred_crop.nii.gz"
    nib.Nifti1Image(expected_1donut_pred_data, np.eye(4)).to_filename(
        expected_pred_path,
    )

    # now crop with crop_gt_and_pred()
    roi_txt_path = "test_assets/box_metrics/test_crop_gt_and_pred.txt"
    gt = sitk.ReadImage(TESTDIR_3D / "shape_5x7x9_3D_1donut.nii.gz")
    pred = sitk.ReadImage(TESTDIR_3D / "shape_5x7x9_3D_1donut_multiclass.nii.gz")
    cropped_gt, cropped_pred = crop_gt_and_pred(roi_txt_path, gt, pred)

    assert np.array_equal(
        # NOTE: SimpleITK npy axis ordering is (z,y,x)!
        # reorder from (z,y,x) to (x,y,z)
        sitk.GetArrayFromImage(cropped_gt).transpose((2, 1, 0)),
        nib.load(expected_gt_path).get_fdata(),
    )

    assert np.array_equal(
        sitk.GetArrayFromImage(cropped_pred).transpose((2, 1, 0)),
        nib.load(expected_pred_path).get_fdata(),
    )
