"""
End-to-end test for the entire evaluation pipeline
"""

import json
from pathlib import Path

from topcow24_eval.constants import TASK, TRACK
from topcow24_eval.evaluation import TopCoWEvaluation

TESTDIR = Path("test_assets/")


def test_e2e_TopCoWEvaluation_Task_3_Edg():
    """
    under test_assets/, there are 2 files for
    test-predictions, test-gt
        "gt_fname": {
            "0": "cow-ant-post-classification_gt.JSON",
            "1": "topcow_ct_003.yml"
        },
        "pred_fname": {
            "0": "cow-ant-post-classification.yml",
            "1": "cow-ant-post-classification.json"
        }
        ...
    in:
    - ./test_assets/task_3_edg_predictions
    - ./test_assets/task_3_edg_ground-truth

    There is also an expected metrics.json output in:
    - ./test_assets/task_3_edge_output

    num_input_pred = 2
    num_ground_truth = 2
    task_3_edg_predictions/
    ├── graph_0/
    │   └── output/
    │       └── cow-ant-post-classification.yml
    └── graph_1/
        └── output/
            └── cow-ant-post-classification.json
    task_3_edg_ground-truth/
    ├── cow-ant-post-classification_gt.JSON
    └── topcow_ct_003.yml

    This test runs TopCoWEvaluation().evaluate() for Task-3-CoW-Classification
    and compares the metrics.json with the expected metrics.json.

    This is the final e2e test that the whole pipeline must pass
    for Task-3-CoW-Classification
    """

    track = TRACK.MR
    task = TASK.GRAPH_CLASSIFICATION
    expected_num_cases = 2
    need_crop = True  # irrelevant

    # folder prefix to differentiate the tasks
    prefix = "task_3_edg_"

    # output_path for clean up
    output_path = f"{prefix}output_test_e2e_TopCoWEvaluation_Task_3_Edg/"
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
