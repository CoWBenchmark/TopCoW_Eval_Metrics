"""
End-to-end test for the entire evaluation pipeline
"""

import json
from pathlib import Path

from topcow24_eval.constants import TASK, TRACK
from topcow24_eval.evaluation import TopCoWEvaluation

TESTDIR = Path("test_assets/")


def test_e2e_TopCoWEvaluation_Task_1_Seg_need_crop():
    """
    under test_assets/, there are two files for
    test-predictions, test-gt, and test-roi-metadata
        "gt_fname": {
            "0": "shape_5x7x9_3D_1donut.nii.gz",
            "1": "shape_8x8x8_3D_8Cubes_gt.nii.gz"
        },
        "pred_fname": {
            "0": "shape_5x7x9_3D_1donut_multiclass.mha",
            "1": "shape_8x8x8_3D_8Cubes_pred.mha"
        }
        ...
    in:
    - ./test_assets/task_1_seg_predictions
    - ./test_assets/task_1_seg_ground-truth
    - ./test_assets/task_1_seg_roi-metadata

    There is also an expected metrics.json output in:
    - ./test_assets/task_1_seg_output

    with need_crop=True, i.e. using crop and roi-txt files:
        num_input_pred = 2
        num_ground_truth = 2
        num_roi_txt = 2
        task_1_seg_predictions/
        ├── shape_5x7x9_3D_1donut_multiclass.mha
        └── shape_8x8x8_3D_8Cubes_pred.mha
        task_1_seg_ground-truth/
        ├── shape_5x7x9_3D_1donut.nii.gz
        └── shape_8x8x8_3D_8Cubes_gt.nii.gz
        task_1_seg_roi-metadata/
        ├── shap_5x7x9_3D_1donut_.txt
        └── shape_8x8x8_3D_8Cubes.txt
    (although the roi-txt files are super big, cropping==original image)

    This test runs TopCoWEvaluation().evaluate() for Task-1-CoW-Segmentation
    and compares the metrics.json with the expected metrics.json.

    This is the final e2e test that the whole pipeline must pass
    for Task-1-CoW-Segmentation

    NOTE: Gotcha for label-2 HD/HD95 in shape_8x8x8_3D_8Cubes_pred
    the HD/HD95 is 0 even though gt and pred are different
    (the segmentation surface contour is the same...).

    This is similar to example from Fig 59 of
    Common Limitations of Image Processing Metrics: A Picture Story
    with hole inside Pred 2 bigger hole.
    see test_hd95_single_label_Fig59_hole() from test_cls_avg_hd95.py
    """

    track = TRACK.CT
    task = TASK.MULTICLASS_SEGMENTATION
    expected_num_cases = 2
    need_crop = True

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = f"{prefix}output_test_e2e_TopCoWEvaluation_CT_need_crop/"
    output_path = Path(output_path)

    evalRun = TopCoWEvaluation(
        track,
        task,
        expected_num_cases,
        need_crop=need_crop,
        predictions_path=TESTDIR / f"{prefix}predictions/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth/",
        output_path=output_path,
        roi_path=TESTDIR / f"{prefix}roi-metadata/",
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


def test_e2e_TopCoWEvaluation_Task_1_Seg_no_crop():
    """
    test_e2e_TopCoWEvaluation_Task_1_Seg with or without crop
    should give the same metrics.json results
    (because the crop roi-txt files are much larger than the images)
    """
    # NOTE: changing the track to MR from CT should also affect nothing!
    track = TRACK.MR
    task = TASK.MULTICLASS_SEGMENTATION
    expected_num_cases = 2
    need_crop = False  # no crop!

    # folder prefix to differentiate the tasks
    prefix = "task_1_seg_"

    # output_path for clean up
    output_path = f"{prefix}output_test_e2e_TopCoWEvaluation_MR_no_crop/"
    output_path = Path(output_path)

    evalRun = TopCoWEvaluation(
        track,
        task,
        expected_num_cases,
        need_crop=need_crop,
        predictions_path=TESTDIR / f"{prefix}predictions/",
        ground_truth_path=TESTDIR / f"{prefix}ground-truth/",
        output_path=output_path,
        # no roi_path supplied, but even if roi_path supplied
        # since need_crop is False, it will be ignored
    )

    # run the evaluation
    evalRun.evaluate()

    # compare the two metrics.json

    # the same expected metrics.json as
    # from test_e2e_TopCoWEvaluation_Task_1_Seg_need_crop()
    with open(TESTDIR / f"{prefix}output/expected_e2e_test_metrics.json") as f:
        expected_metrics_json = json.load(f)

    # no crop run results
    with open(output_path / "metrics.json") as f:
        generated_metrics_json = json.load(f)

    assert expected_metrics_json == generated_metrics_json

    print(f"expected_metrics_json =\n{json.dumps(expected_metrics_json, indent=2)}")
    print(f"generated_metrics_json =\n{json.dumps(generated_metrics_json, indent=2)}")

    # clean up the new metrics.json
    (output_path / "metrics.json").unlink()
    # clean up the output_path folder
    output_path.rmdir()
