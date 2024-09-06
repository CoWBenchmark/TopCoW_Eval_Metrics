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
        size_arr = re.findall(r"\b\d+\b", lines[1])
        location_arr = re.findall(r"\b\d+\b", lines[2])

    print(lines)
    print(size_arr)
    print(location_arr)
    print("--- EOF roi_txt ---")

    return ([int(x) for x in size_arr], [int(x) for x in location_arr])


def parse_roi_json(roi_json) -> tuple[list, list]:
    """
    similart to parse_roi_txt() but for GC's cow-roi.json
    parse the ROI size and location from cow-roi.json

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
