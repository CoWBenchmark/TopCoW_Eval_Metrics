"""
utility functions to work with nibael, mha and SimpleITK
"""
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk


def convert_mha_nii(img_path, file_ending, save_path="."):
    """
    convert between .mha metaImage to .nii.gz nifti compressed
    file_ending='.nii.gz' | '.mha'
    """
    img = sitk.ReadImage(img_path)
    source_fname = os.path.basename(img_path)
    target_fname = source_fname.split(".")[0] + file_ending
    sitk.WriteImage(img, os.path.join(save_path, target_fname), useCompression=True)


def make_dummy_mask(filename, shape=(3, 3, 3)):
    """
    save a dummy mask of all zeros in nii.gz format
    see https://nipy.org/nibabel/gettingstarted.html
    """
    mask = np.zeros(shape, dtype=np.uint8)

    print(f"mask.shape = {mask.shape}")

    img = nib.Nifti1Image(mask, affine=np.eye(4))
    img.to_filename(f"{filename}.nii.gz")

    # DONE!
    print(f"{filename}.nii.gz SAVED!\n")


def load_image_and_array_as_uint8(path):
    """
    Loads segmentation image (nifti or mha) from path,
    cast it to uint8 and returns the SimpleITK.Image and np array
    """
    print(f"\n-- call load_image_and_array_as_uint8({path})")
    # read image
    img = sitk.ReadImage(path)
    access_sitk_attr(img)

    # Cast to the same type
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkUInt8)
    caster.SetNumberOfThreads(1)
    img = caster.Execute(img)

    # NOTE: Axis order is (z,y,x)
    arr = sitk.GetArrayFromImage(img)
    # reorder from (z,y,x) to (x,y,z)
    arr = arr.transpose((2, 1, 0)).astype(np.uint8)
    print("load_image_and_array_as_uint8 arr.shape = ", arr.shape)
    return img, arr


def access_sitk_attr(image: sitk.Image):
    """
    Accessing Attributes
    """
    print("##############################################")
    print(f"image.GetSize() = {image.GetSize()}")
    print(f"image.GetWidth() = {image.GetWidth()}")
    print(f"image.GetHeight() = {image.GetHeight()}")
    print(f"image.GetDepth() = {image.GetDepth()}")
    print(f"image.GetOrigin() = {image.GetOrigin()}")
    print(f"image.GetSpacing() = {image.GetSpacing()}")
    print(f"image.GetDirection() = {image.GetDirection()}")
    print("image.GetNumberOfComponentsPerPixel() =")
    print(image.GetNumberOfComponentsPerPixel())
    print(f"image.GetDimension() = {image.GetDimension()}")
    print(f"image.GetPixelIDValue() = {image.GetPixelIDValue()}")
    print(f"image.GetPixelIDTypeAsString() = {image.GetPixelIDTypeAsString()}")
    for key in image.GetMetaDataKeys():
        print('"{0}":"{1}"'.format(key, image.GetMetaData(key)))
    print("##############################################")


if __name__ == "__main__":
    make_dummy_mask("shape_5x5_2D", (5, 5, 1))
    make_dummy_mask("shape_6x3_2D", (6, 3, 1))
    print("utils.")
