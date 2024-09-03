# TopCoW Evaluation Metrics 🐮

This repo contains the package to compute the evaluation metrics for the [**TopCoW2024 challenge**](https://topcow24.grand-challenge.org/) on grand-challnge (GC).

## `topcow24_eval` package

At the root folder, there is a [`pyproject.toml`](./pyproject.toml) config file that can set up the evaluation project folder
as a local pip module called **`topcow24_eval`** for running the evaluations in your python project.

To setup and install `topcow24_eval` package:

```sh
# from topcow24_eval root

# use py3.10
python3.10 -m venv env_py310
source env_py310/bin/activate

pip install -e ".[dev]"
```

### Segmentation metrics

In [`./topcow24_eval/metrics/seg_metrics/`](./topcow24_eval/metrics/seg_metrics/), you will find our implementations for evaluating the submitted segmentation predictions.

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
6. Variant-balanced graph classification accuracy:
    * [`graph_classification.py`](./topcow24_eval/metrics/seg_metrics/graph_classification/graph_classification.py)
7. Variant-balanced topology match rate:
    * [`topology_matching.py`](./topcow24_eval/metrics/seg_metrics/topology_matching/topology_matching.py)

### Bounding box metrics

Coming soon

### Graph Classification metrics

Coming soon

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
* topology matching:
    * [`test_topology_matching.py](./topcow24_eval/metrics/seg_metrics/topology_matching/test_topology_matching.py)

Test asset files used in the test cases are stored in the folder [`./test_assets/`](./test_assets/).

Simply invoke the tests by `pytest .`:

```bash
# simply run pytest
$ pytest .

topcow24_eval/metrics/seg_metrics/graph_classification/test_edge_criteria.py ..                                                                        [  2%]
topcow24_eval/metrics/seg_metrics/graph_classification/test_generate_edgelist.py ...                                                                   [  6%]
topcow24_eval/metrics/seg_metrics/graph_classification/test_graph_classification.py .                                                                  [  7%]
topcow24_eval/metrics/seg_metrics/test_clDice.py ....                                                                                                  [ 12%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_b0.py ...............                                                                                   [ 31%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_dice.py ...........                                                                                     [ 45%]
topcow24_eval/metrics/seg_metrics/test_cls_avg_hd95.py .............                                                                                   [ 62%]
topcow24_eval/metrics/seg_metrics/test_detection_grp2_labels.py ......                                                                                 [ 69%]
topcow24_eval/metrics/seg_metrics/test_generate_cls_avg_dict.py ...........                                                                            [ 83%]
topcow24_eval/metrics/seg_metrics/topology_matching/test_topology_matching.py ......                                                                   [ 91%]
topcow24_eval/utils/test_utils_mask.py .....                                                                                                           [ 97%]
topcow24_eval/utils/test_utils_neighborhood.py ..                                                                                                      [100%]

======================================= 80 passed, 7 warnings in 7.15s =======================================
```
