"""
utility functions to work with nibael, mha and SimpleITK
"""

import numpy as np
import SimpleITK as sitk


def load_image_and_array_as_uint8(
    path, log_sitk_attr=False
) -> tuple[sitk.Image, np.ndarray]:
    """
    Loads segmentation image (nifti or mha) from path,
    cast it to uint8 and returns the SimpleITK.Image and np array
    """
    print(f"\n-- call load_image_and_array_as_uint8({path})")
    # read image
    img = sitk.ReadImage(path)

    if log_sitk_attr:
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
    # for key in image.GetMetaDataKeys():
    #     print('"{0}":"{1}"'.format(key, image.GetMetaData(key)))
    print("##############################################")
