# TopCoW Evaluation Metrics 🐮

This repo contains the evaluation metric functions used for the [**TopCoW2023 challenge**](https://topcow23.grand-challenge.org/) on grand-challnge (GC).

## `metric_functions.py`

In [`./metric_functions.py`](./metric_functions.py), you will find our implementations for evaluating the submitted segmentation predictions.
We assess the metrics such as Dice coefficient and cl-Dice.

## Unit-test with `test_metric_functions.py`

The documentations for our code come in the form of unit-tests.
The file [`./test_metric_functions.py`](./test_metric_functions.py) contains the test cases for the evaluation metrics.
Nifti files used in the test cases are stored in the folder [`./test_metrics/`](./test_metrics/).

```bash
# simply run pytest
$ pytest test_metric_functions.py 
=============================================================== test session starts ================================================================
platform linux -- Python 3.11.4, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/svd/Documents/TopCoWEvaluation
plugins: anyio-3.7.1
collected 8 items                                                                                                                                  

test_metric_functions.py ........                                                                                                            [100%]

================================================================ 8 passed in 0.40s =================================================================
```
