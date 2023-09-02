# TopCoW Evaluation Metrics 🐮

This repo contains the evaluation metric functions used for the [**TopCoW2023 challenge**](https://topcow23.grand-challenge.org/) on grand-challnge (GC).

## `metric_functions.py`

In [`./metric_functions.py`](./metric_functions.py), you will find our implementations for evaluating the submitted segmentation predictions.
We assess the metrics such as Dice coefficient and cl-Dice.

## Unit-test for `test_*.py`

The documentations for our code come in the form of unit-tests.
The files with names that follow the form `test_*.py` contain the test cases for the evaluation metrics.

* Dice: [`./test_Dice_dict.py`](./test_Dice_dict.py)
* clDice: [`./test_clDice.py`](./test_clDice.py)
* Betti number error: [`./test_betti_num.py`](./test_betti_num.py)

Nifti files used in the test cases are stored in the folder [`./test_metrics/`](./test_metrics/).

Simply invode the tests by `pytest .`:

```bash
# simply run pytest
$ pytest .
============================================ test session starts =============================================
platform linux -- Python 3.11.4, pytest-7.4.0, pluggy-1.2.0
rootdir: /home/svd/Documents/TopCoWEvaluation
plugins: anyio-3.7.1
collected 23 items                                                                                           

test_Dice_dict.py ........                                                                             [ 34%]
test_betti_num.py ............                                                                         [ 86%]
test_clDice.py ...                                                                                     [100%]

============================================= 23 passed in 0.50s =============================================
```
