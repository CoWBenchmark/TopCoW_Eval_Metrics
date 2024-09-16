"""
End-to-end test for the entire evaluation pipeline
"""

import json
from pathlib import Path

from topcow24_eval.constants import TASK, TRACK
from topcow24_eval.evaluation import TopCoWEvaluation

TESTDIR = Path("test_assets/")


def test_e2e_TopCoWEvaluation_Task_2_Box():
    """
    under test_assets/, there are 3 files for
    test-predictions, test-gt
        "gt_fname": {
            "0": "test_iou_dict_from_files_hollow_6x3x3_4x3x3_box_2.TXT",
            "1": "test_iou_dict_from_files_hollow_6x3x3_shifted_box_2.txt",
            "2": "test_iou_dict_from_files_tiny_vs_small_box_2.txt"
        },
        "pred_fname": {
            "0": "cow-roi.json",
            "1": "cow-roi.json",
            "2": "cow-roi.json"
        }
        ...
    in:
    - ./test_assets/task_2_box_predictions
    - ./test_assets/task_2_box_ground-truth

    There is also an expected metrics.json output in:
    - ./test_assets/task_2_box_output

    num_input_pred = 3
    num_ground_truth = 3
    task_2_box_predictions/
    ├── hollow_6x3x3_4x3x3/
    │   └── output/
    │       └── cow-roi.json
    ├── hollow_6x3x3_shifted/
    │   └── output/
    │       └── cow-roi.json
    └── tiny_vs_small/
        └── output/
            └── cow-roi.json
    task_2_box_ground-truth/
    ├── test_iou_dict_from_files_hollow_6x3x3_4x3x3_box_2.TXT
    ├── test_iou_dict_from_files_hollow_6x3x3_shifted_box_2.txt
    └── test_iou_dict_from_files_tiny_vs_small_box_2.txt

    This test runs TopCoWEvaluation().evaluate() for Task-2-CoW-ObjDet
    and compares the metrics.json with the expected metrics.json.

    The 3 iou pairs:
    0. 6x3x3_4x3x3:
        boundary_iou == 34 / 52  -> 0.653
        "IoU": 36 / 54,          -> 0.666
    1. 6x3x3_shifted:
        boundary_iou == 10 / 94  -> 0.106
        "IoU": 12 / 96,          -> 0.125
    2. tiny_vs_small:
        boundary_iou == 6 / 34   -> 0.176
        "IoU": 6 / 36,           -> 0.166

    mean boundary_iou = 0.312
    mean iou = 0.319

    This is the final e2e test that the whole pipeline must pass
    for Task-2-CoW-ObjDet
    """

    track = TRACK.CT
    task = TASK.OBJECT_DETECTION
    expected_num_cases = 3
    need_crop = True  # irrelevant

    # folder prefix to differentiate the tasks
    prefix = "task_2_box_"

    # output_path for clean up
    output_path = f"{prefix}output_test_e2e_TopCoWEvaluation_Task_2_Box/"
    output_path = Path(output_path)

    evalRun = TopCoWEvaluation(
        track,
        task,
        expected_num_cases,
        need_crop=need_crop,
        predictions_path=TESTDIR / f"{prefix}predictions/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth/",
        output_path=output_path,
    )

    # run the evaluation
    evalRun.evaluate()

    # compare the two metrics.json

    with open(TESTDIR / f"{prefix}output/expected_e2e_test_metrics.json") as f:
        expected_metrics_json = json.load(f)

    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()
