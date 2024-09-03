import re


def parse_roi_txt(roi_txt):
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