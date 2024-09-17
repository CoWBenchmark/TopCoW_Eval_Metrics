# TopCoW Evaluation Metrics 🐮

This repo contains the package to compute the evaluation metrics for the [**TopCoW2024 challenge**](https://topcow24.grand-challenge.org/) on grand-challnge (GC).

## `topcow24_eval` package

At the root folder, there is a [`pyproject.toml`](./pyproject.toml) config file that can set up the evaluation project folder
as a local pip module called **`topcow24_eval`** for running the evaluations in your python project.

To setup and install `topcow24_eval` package:

```sh
# from topcow24_eval root
bash ./setup.sh

# activate the env with topcow24_eval installed
source env_py310/bin/activate
```

## Run Evaluations with `python3 topcow24_eval/evaluation.py`

### 1. Configure with `configs.py`

First go to `topcow24_eval/configs.py` and configure the track, task, and `expected_num_cases`.
The `expected_num_cases` is required and must match the number of cases to evalute, i.e. the number of ground-truth cases etc.
See below.

### 2. Folders `ground-truth/` and `predictions/`

When not in docker environment, the paths of pred, gt, roi etc
are set by default to be on the same level as the package dir `topcow24_eval`:

```sh
# mkdir and put your gt, pred etc like this:
├── ground-truth
├── predictions
├── topcow24_eval
```

Simply put the files of ground-truth and predictions in the folders `ground-truth/` and `predictions/`,
and run `python3 topcow24_eval/evaluation.py`.

_You can also specify your own custom paths for the ground-truth, predictions etc when you call the evaluation object:_

```py
# example from topcow24_eval/test_evaluation.py
    evalRun = TopCoWEvaluation(
        track,
        task,
        expected_num_cases,
        need_crop,
        predictions_path=TESTDIR / "task_1_seg_predictions/",
        ground_truth_path=TESTDIR / "task_1_seg_ground-truth/",
        output_path=output_path,
        roi_path=TESTDIR / "task_1_seg_roi-metadata/",
    )
```

**The naming of gt and pred files can be arbitrary as long as their filelist dataframe `.sort_values()` are sorted in the same way!**

The accepted file formats for ground-truth and predictions are:

- NIfTI (`.nii.gz`, `.nii`) or SimpleITK compatible images `.mha` for images and masks
- `.txt`, `.json` for bounding box
    - (_`roi-metadata/` only allows for `.txt` for roi-txt. See below._)
- `.yml`, `.json` for graph/edge-list

#### 2.1 Folder of `roi-metadata/` for Task-1-CoW-Segmentation

Optionally, if you evaluate for Task-1-CoW-Segmentation, you can decide to whether evaluate on the
cropped region (ROI) of the ground-truth/prediction.

Whether to crop the images for evaluations is set by `need_crop` in `configs.py`.

If `need_crop` is `False`, then `roi_path` will be ignored, and no cropping will be done.
If `need_crop` is `True` and `roi_path` has roi_txt files,
then the evaluations will be performed on the cropped gt, pred.
It has no effect on Task 2 or Task 3.

Afterwards, make sure to put the roi-txt files in the folder `roi-metadata/` (or you can supply your own `roi_path` when calling `TopCoWEvaluation()`):

```sh
# note the new roi-metadata/
├── ground-truth
├── predictions
├── roi-metadata
├── topcow24_eval
```

**The naming of roi-txt files can be arbitrary as long as their filelist dataframe `.sort_values()` are sorted in the same way as gt or pred!**

---

### Segmentation metrics

In [`topcow24_eval/metrics/seg_metrics/`](./topcow24_eval/metrics/seg_metrics/), you will find our implementations for evaluating the submitted segmentation predictions.

Seven evaluation metrics with equal weights for multi-class (CoW anatomical vessels) segmentation task:

1. Class-average Dice similarity coefficient:
    * [`cls_avg_dice.py`](./topcow24_eval/metrics/seg_metrics/cls_avg_dice.py)
2. Centerline Dice (clDice) on merged binary mask:
    * [`clDice.py`](./topcow24_eval/metrics/seg_metrics/clDice.py)
3. Class-average 0-th Betti number error:
    * [`cls_avg_b0.py`](./topcow24_eval/metrics/seg_metrics/cls_avg_b0.py)
4. Class-average Hausdorff Distance 95% Percentile (HD95):
    * [`cls_avg_hd95.py`](./topcow24_eval/metrics/seg_metrics/cls_avg_hd95.py)
5. Average F1 score (harmonic mean of the precision and recall) for detection of the "Group 2 CoW components":
    * [`detection_grp2_labels.py`](./topcow24_eval/metrics/seg_metrics/detection_grp2_labels.py)
    * [`aggregate_all_detection_dicts.py`](./topcow24_eval/aggregate/aggregate_all_detection_dicts.py)
6. Variant-balanced graph classification accuracy:
    * [`graph_classification.py`](./topcow24_eval/metrics/seg_metrics/graph_classification/graph_classification.py)
    * [`aggregate_all_graph_dicts.py`](./topcow24_eval/aggregate/aggregate_all_graph_dicts.py)
7. Variant-balanced topology match rate:
    * [`topology_matching.py`](./topcow24_eval/metrics/seg_metrics/topology_matching/topology_matching.py)
    * [`aggregate_all_topo_dicts.py`](./topcow24_eval/aggregate/aggregate_all_topo_dicts.py)

### Bounding box metrics

In [`topcow24_eval/metrics/box_metrics/`](./topcow24_eval/metrics/box_metrics/), you will find our implementations for evaluating bounding box predictions.

1. Boundary Intersection over Union (IoU) and IoU:
    * [`iou_dict_from_files.py`](./topcow24_eval/metrics/box_metrics/iou_dict_from_files.py)

### Graph Classification metrics

In [`topcow24_eval/metrics/edg_metrics/`](./topcow24_eval/metrics/edg_metrics/), you will find our implementations for evaluating graph classification task.

1. Variant-balanced accuracy:
    * [`graph_dict_from_files.py`](./topcow24_eval/metrics/edg_metrics/graph_dict_from_files.py)
    * [`aggregate_all_graph_dicts.py`](./topcow24_eval/aggregate/aggregate_all_graph_dicts.py)
2. Distance between ground-truth and predicted 4-element vectors
    * [`graph_dict_from_files.py`](./topcow24_eval/metrics/edg_metrics/graph_dict_from_files.py)

---

## Unit tests as documentation

The documentations for our code come in the form of unit tests.
Please check our test cases to see the expected inputs and outputs, expected behaviors and calculations.

The files with names that follow the form `test_*.py` contain the test cases for the evaluation metrics.

* Dice:
    * [`test_cls_avg_dice.py`](./topcow24_eval/metrics/seg_metrics/test_cls_avg_dice.py)
* clDice:
    * [`test_clDice.py`](./topcow24_eval/metrics/seg_metrics/test_clDice.py)
* Betti-0 number error:
    * [`test_cls_avg_b0.py`](./topcow24_eval/metrics/seg_metrics/test_cls_avg_b0.py)
* HD and HD95:
    * [`test_cls_avg_hd95.py`](./topcow24_eval/metrics/seg_metrics/test_cls_avg_hd95.py)
* graph classification:
    * [`test_graph_classification.py`](./topcow24_eval/metrics/seg_metrics/graph_classification/test_graph_classification.py)
    * [`test_graph_dict_from_files.py`](./topcow24_eval/metrics/edg_metrics/test_graph_dict_from_files.py)
    * [`test_aggregate_all_graph_dicts.py`](./topcow24_eval/aggregate/test_aggregate_all_graph_dicts.py)
* detections:
    * [`test_aggregate_all_detection_dicts.py`](./topcow24_eval/aggregate/test_aggregate_all_detection_dicts.py)
* topology matching:
    * [`test_topology_matching.py`](./topcow24_eval/metrics/seg_metrics/topology_matching/test_topology_matching.py)
    * [`test_aggregate_all_topo_dicts.py`](./topcow24_eval/aggregate/test_aggregate_all_topo_dicts.py)
* boundary IoU:
    * [`test_iou_dict_from_files.py`](./topcow24_eval/metrics/box_metrics/test_iou_dict_from_files.py)

Test asset files used in the test cases are stored in the folder [`test_assets/`](./test_assets/).

Simply invoke the tests by `pytest .`:

```bash
# simply run pytest
$ pytest .

topcow24_eval/aggregate/test_aggregate_all_detection_dicts.py ...                                                                            [  2%]
topcow24_eval/aggregate/test_aggregate_all_graph_dicts.py ...                                                                                [  4%]
topcow24_eval/aggregate/test_aggregate_all_topo_dicts.py .                                                                                   [  4%]
topcow24_eval/aggregate/test_edge_list_to_variant_str.py .                                                                                   [  5%]
topcow24_eval/metrics/box_metrics/test_boundary_iou_from_tuple.py ...........                                                                [ 13%]
topcow24_eval/metrics/box_metrics/test_boundary_points_with_distances.py .....                                                               [ 16%]
topcow24_eval/metrics/box_metrics/test_iou_dict_from_files.py ....                                                                           [ 19%]
topcow24_eval/metrics/edg_metrics/test_edge_dict_to_list.py ..                                                                               [ 20%]
topcow24_eval/metrics/edg_metrics/test_graph_dict_from_files.py ..                                                                           [ 22%]
topcow24_eval/metrics/seg_metrics/graph_classification/test_edge_criteria.py ..                                                              [ 23%]
topcow24_eval/metrics/seg_metrics/graph_classification/test_generate_edgelist.py ...                                                         [ 25%]
topcow24_eval/metrics/seg_metrics/graph_classification/test_graph_classification.py .                                                        [ 26%]
topcow24_eval/metrics/seg_metrics/test_clDice.py ....                                                                                        [ 29%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_b0.py ...............                                                                         [ 39%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_dice.py .............                                                                         [ 48%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_hd95.py .............                                                                         [ 57%]
topcow24_eval/metrics/seg_metrics/test_detection_grp2_labels.py ........                                                                     [ 63%]
topcow24_eval/metrics/seg_metrics/test_generate_cls_avg_dict.py ..........                                                                   [ 70%]
topcow24_eval/metrics/seg_metrics/topology_matching/test_check_LR_flip.py ..                                                                 [ 71%]
topcow24_eval/metrics/seg_metrics/topology_matching/test_topology_matching.py ........                                                       [ 77%]
topcow24_eval/test_evaluation_task_1_seg.py ..                                                                                               [ 78%]
topcow24_eval/test_evaluation_task_2_box.py .                                                                                                [ 79%]
topcow24_eval/test_evaluation_task_3_edg.py .                                                                                                [ 79%]
topcow24_eval/test_score_case.py ....                                                                                                        [ 82%]
topcow24_eval/utils/test_crop_gt_and_pred.py .                                                                                               [ 83%]
topcow24_eval/utils/test_crop_sitk.py ....                                                                                                   [ 86%]
topcow24_eval/utils/test_utils_box.py ..........                                                                                             [ 93%]
topcow24_eval/utils/test_utils_edge.py ..                                                                                                    [ 94%]
topcow24_eval/utils/test_utils_mask.py ......                                                                                                [ 98%]
topcow24_eval/utils/test_utils_neighborhood.py ..                                                                                            [100%]

======================================================== 144 passed, 15 warnings in 10.54s =========================================================
```
