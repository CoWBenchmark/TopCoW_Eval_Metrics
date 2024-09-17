"""
for Grand-challenge.org (GC) docker environment

adapted from GC challenge pack example evaluation
https://github.com/DIAGNijmegen/demo-challenge-pack

and from older challenges
https://github.com/DIAGNijmegen/drive-evaluation
"""

import json
from glob import glob
from pathlib import Path

from topcow24_eval.constants import TASK


def get_image_name(*, values: list[dict], slug: str) -> str:
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values: list[dict], slug: str) -> str:
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(
    *, job_pk: str, values: list[dict], slug: str, input_dir: Path
) -> Path:
    """
    NOTE: task-1's relative_path is a directory
    "relative_path": "images/cow-multiclass-segmentation"

    BUT for task-2 and task-3 they are files!!?!
    task-2: "relative_path": "cow-roi.json"
    task-3: "relative_path": "cow-ant-post-classification.json"

    > Task 1: Circle of Willis Multi-class Segmentation (Segmentation) to
    **`/output/images/cow-multiclass-segmentation/<uuid>.mha`**

    > Task 2: Circle of Willis Bounding Box (Json) to
    **`/output/cow-roi.json`**

    > Task 3: Circle of Willis Edge Classification (Json) to
    **`/output/cow-ant-post-classification.json`**

    for example:
    input/
    ├── 08024e28-4016-4984-a5a0-5de5264d8037/
    │   └── output/
    │       └── images/
    │           └── cow-multiclass-segmentation/
    │               └── 40ee5b24-ec04-4153-b1ba-7e5bda52cbc3.mha
    """
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return input_dir / job_pk / "output" / relative_path


def load_predictions_json(
    fname: Path, slug_input: str, slug_output: str, task: TASK
) -> dict:
    """
    "predictions.json" contains the location of the submitted algorithm's
    predictions generated by jobs on GC cloud.
    This json tells us how the output filenames
    are mapped to the input filenames.

    code adapted from 2023 version of
    https://grand-challenge.org/documentation/automated-evaluation/
    which is from
    https://github.com/DIAGNijmegen/drive-evaluation/blob/master/jsonloader.py

    schema is [{pk}, {pk}, ...]
    [
        {pk, inputs[], outputs[]},
        {pk, inputs[], outputs[]},
        ...
    ]

    Returns:
    a mapping_dict
    e.g.
    {
        '64db4c4e-a91e-4882-b1b6-8ee5784961dc.mha': 'topcow_ct_whole_091_0000.nii.gz',
        '649d467f-9a70-4834-8d1f-36cb837338c7.mha': 'topcow_ct_whole_092_0000.nii.gz',
        ...
    }
    """
    print(f"\n-- call load_predictions_json(fname={fname})")
    cases = {}

    with open(fname, "r") as f:
        entries = json.load(f)

    if isinstance(entries, float):
        raise TypeError(f"entries of type float for file: {fname}")

    for job in entries:
        # retrieve the image name from input interface to
        # match it with an image in your ground truth
        name = get_image_name(
            values=job["inputs"],
            slug=slug_input,
        )
        print("*** name = ", name)
        # this is the name of the input to the submitted algorithm
        # not the actual ground-truth file.
        # here we rely on the sorting to be the same between
        # inputs and ground-truth files
        # e.g. the _0000 file is made up by the mapping_dict
        # /opt/app/ground-truth/topcow_ct_whole_091_0000.nii.gz
        # vs
        # /opt/app/ground-truth/mul_label_topcow_ct_whole_091.nii.gz
        # or even .yml and .txt ground-truth files
        # -------------------------

        # from output interface
        # find the location of the results
        location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug=slug_output,
        )
        # read the singular result from location
        if task is TASK.MULTICLASS_SEGMENTATION:
            result = glob(str(location / "*.mha"))[0]
        else:
            # task-2 and task-3 output same as relative_path
            result = location
        print("*** result = ", result)

        cases[result] = name

    return cases


def is_docker():
    """
    check if process.py is run in a docker env
        bash test.sh vs python3 process.py

    from https://stackoverflow.com/questions/43878953/how-does-one-detect-if-one-is-running-within-a-docker-container-within-python
    """
    cgroup = Path("/proc/self/cgroup")
    exec_in_docker = (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and "docker" in cgroup.read_text()
    )
    print(f"exec_in_docker? {exec_in_docker}")
    return exec_in_docker