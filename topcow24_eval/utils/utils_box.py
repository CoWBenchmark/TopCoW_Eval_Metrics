import json
import re


def parse_roi_txt(roi_txt) -> tuple[list, list]:
    """
    parse the ROI metadata txt file

    input:
        roi_txt
            meta text file_path containing the ROI info
            Size = number of pixels along the x, y, z axis
            Location = coordinate of the x-min, y-min, z-min (0-indexed)

    output:
        size_arr, location_arr
    """
    print(f"\n--- parse_roi_txt({roi_txt}) ---")
    with open(roi_txt) as f:
        lines = f.readlines()

    print(lines)

    size_arr = re.findall(r"\b\d+\b", lines[1])
    location_arr = re.findall(r"\b\d+\b", lines[2])

    print(size_arr)
    print(location_arr)
    print("--- EOF roi_txt ---")

    return ([int(x) for x in size_arr], [int(x) for x in location_arr])


def parse_roi_json(roi_json) -> tuple[list, list]:
    """
    similar to parse_roi_txt() but for GC's cow-roi.json
    parse the ROI size and location from cow-roi.json

    The "size" is an array of number of voxels of the volume
    along the x, y, and z axis.

    The "location" is an array for the lower coordinates
    of x-min, y-min, z-min (0-indexed) of the 3D volume.

    example json:
        {"size": [70, 61, 17], "location": [35, 30, 8]}
    """
    print(f"\n--- parse_roi_json({roi_json}) ---")
    with open(roi_json, mode="r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"json data = {data}")

    size_arr = data["size"]
    location_arr = data["location"]

    print(size_arr)
    print(location_arr)
    print("--- EOF roi_json ---")

    return ([int(x) for x in size_arr], [int(x) for x in location_arr])


def get_end_index(start_index, dim_size) -> int:
    """
    # x_end = x_start + x_size - 1

    returns the end_index of the slice

    E.g. [ABCDE], start_index = 1, size = 2
    end slice is C, end_index = 1+2-1 = 2 (index of C)
    """
    return int(start_index + dim_size - 1)


def get_size(start_index, end_index) -> int:
    # reverse of get_end_index
    return int(end_index + 1 - start_index)


def get_dcm_slice(img_shape, size_arr, location_arr, return_correction=False):
    """
    input:
        img_shape of the image to be sliced on
        arrays of sizes and locations
        x = L+, y=P+, z=S+

    output:
        dcm_slice
            slice object based on the start and size
        if slice object outside of the img_shape, correct for error
        if return_correction:
            (corrected) size_arr,
            (corrected) location_arr
    """
    print("img_shape = ", img_shape)

    x_size, y_size, z_size = int(size_arr[0]), int(size_arr[1]), int(size_arr[2])

    # Sizes are only adjusted if start_index is negative OR
    # end_index is out of range
    # Sizes are not adjusted on its own

    dcm_x_start, dcm_y_start, dcm_z_start = (
        int(location_arr[0]),
        int(location_arr[1]),
        int(location_arr[2]),
    )
    if not ((dcm_x_start >= 0) and (dcm_y_start >= 0) and (dcm_z_start >= 0)):
        print("[ALERT!] cropped xyz start < 0")

    dcm_x_start, dcm_y_start, dcm_z_start = (
        max(0, dcm_x_start),
        max(0, dcm_y_start),
        max(0, dcm_z_start),
    )

    dcm_x_end = get_end_index(dcm_x_start, x_size)
    dcm_y_end = get_end_index(dcm_y_start, y_size)
    dcm_z_end = get_end_index(dcm_z_start, z_size)

    if not (
        (dcm_x_end <= img_shape[0] - 1)
        and (dcm_y_end <= img_shape[1] - 1)
        and (dcm_z_end <= img_shape[2] - 1)
    ):
        print("[ALERT!] cropped xyz end out of range")

    dcm_x_end, dcm_y_end, dcm_z_end = (
        min(img_shape[0] - 1, dcm_x_end),
        min(img_shape[1] - 1, dcm_y_end),
        min(img_shape[2] - 1, dcm_z_end),
    )

    # get the new size_arr just in case
    x_size = get_size(dcm_x_start, dcm_x_end)
    y_size = get_size(dcm_y_start, dcm_y_end)
    z_size = get_size(dcm_z_start, dcm_z_end)

    print(
        "dcm_x_start, dcm_y_start, dcm_z_start = ",
        dcm_x_start,
        dcm_y_start,
        dcm_z_start,
    )
    print("x_size, y_size, z_size = ", x_size, y_size, z_size)
    print("dcm_x_end, dcm_y_end, dcm_z_end = ", dcm_x_end, dcm_y_end, dcm_z_end)

    # construct the dicom slice indices obj
    dcm_slice = (
        slice(dcm_x_start, dcm_x_end + 1),
        slice(dcm_y_start, dcm_y_end + 1),
        slice(dcm_z_start, dcm_z_end + 1),
    )

    if return_correction:
        return (
            dcm_slice,
            [int(x_size), int(y_size), int(z_size)],
            [
                int(dcm_x_start),
                int(dcm_y_start),
                int(dcm_z_start),
            ],
        )
    else:
        return dcm_slice
