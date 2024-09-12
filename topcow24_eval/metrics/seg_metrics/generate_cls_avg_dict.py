"""
Helper for generating a class average score dict

Metrics for Task-1-CoW-Segmentation
"""

import pprint
from enum import Enum
from typing import Callable

import numpy as np
import SimpleITK as sitk
from topcow24_eval.constants import (
    BIN_CLASS_LABEL_MAP,
    MUL_CLASS_LABEL_MAP,
)
from topcow24_eval.utils.utils_mask import extract_labels


def generate_cls_avg_dict(
    *,
    gt: sitk.Image,
    pred: sitk.Image,
    task: Enum,
    metric_keys: list[str],
    metric_func: Callable,
) -> dict:
    """
    Parameters:
        gt:
            ground truth sitk.Image
        pred:
            prediction sitk.Image
        task:
            [deprecated] If task is TASK.BINARY_SEGMENTATION,
                it will compute the MergedBin score
            If task is TASK.MULTICLASS_SEGMENTATION, it will
                compute metric scores for union of present classes
                and an overall average per case.
        metric_keys:
            list of metric_keys
            E.g. ["Dice"], ["B0err"], ["HD", "HD95"]...
            For each metric_key will create a:
                {metric_key: metric_score}
        metric_func:
            Callable function that takes in
            func:= (gt, pred, label) => List[metric_scores] for that label

    Returns:
        cls_avg_dict
        NOTE: returned cls_avg_dict only considers all labels (classes)
            that are present in both gt and pred to compute the
            class-average-metric per case
    """
    print("\n-- generate_cls_avg_dict()")
    print(f"task = {task}")
    print(f"metric_keys = {metric_keys}")
    print(f"metric_func = {metric_func.__name__}\n")

    # gt and pred should have the same shape
    assert gt.GetSize() == pred.GetSize(), "gt pred not matching shapes!"

    # img should be in 3D
    assert gt.GetDimension() == 3, "sitk img should be in 3D"

    # NOTE: SimpleITK npy axis ordering is (z,y,x)!
    # reorder from (z,y,x) to (x,y,z)
    gt_array = sitk.GetArrayFromImage(gt).transpose((2, 1, 0)).astype(np.uint8)
    pred_array = sitk.GetArrayFromImage(pred).transpose((2, 1, 0)).astype(np.uint8)

    # SimpleITK image GetSize should have the same shape as transposed (x,y,z) npy
    assert (
        gt.GetSize() == gt_array.shape
    ), f"gt.GetSize():{gt.GetSize()} != gt_array.shape:{gt_array.shape}"
    assert (
        pred.GetSize() == pred_array.shape
    ), f"pred.GetSize():{pred.GetSize()} != pred_array.shape:{pred_array.shape}"

    labels = extract_labels(gt_array, pred_array)

    cls_avg_dict = {}

    # key in cls_avg_dict for class average
    cls_avg_keys = [f"ClsAvg{metric_key}" for metric_key in metric_keys]

    print("### Multiclass Segmentation ###")

    # when there are no labels in the images,
    # return blank cls_avg_dict with only average and merged_binary of 0
    if len(labels) == 0:
        for cls_avg_key in cls_avg_keys:
            # update each class average key to 0
            update_cls_avg_dict(
                cls_avg_dict=cls_avg_dict,
                label=cls_avg_key,
                label_map=None,
                metric_keys=metric_keys,
                metric_scores=[0] * len(metric_keys),
            )

        # update merged binary class to 0
        update_cls_avg_dict(
            cls_avg_dict=cls_avg_dict,
            # use the label_map for merged bin instead of "1"
            label=BIN_CLASS_LABEL_MAP["1"],
            label_map=None,
            metric_keys=metric_keys,
            metric_scores=[0] * len(metric_keys),
        )

        print(f"\ncls_avg_dict = {cls_avg_dict}")
        return cls_avg_dict

    # otherwise compute the metric_scores for
    # all present labels and update the cls_avg_dict

    print("### for all present labels ###")

    sum_scores = np.zeros(len(metric_keys))

    for voxel_label in labels:
        # update the cls_avg_dict for that label
        metric_scores = metric_func(gt=gt, pred=pred, label=voxel_label)

        update_cls_avg_dict(
            cls_avg_dict=cls_avg_dict,
            label=voxel_label,
            label_map=MUL_CLASS_LABEL_MAP,
            metric_keys=metric_keys,
            metric_scores=metric_scores,
        )

        # keep track of the sum of scores for average
        sum_scores += np.array(metric_scores)

    # get the average from sum_scores
    # convert from np array to list
    avg_scores = (sum_scores / len(labels)).tolist()

    # update the class average keys
    for cls_avg_key in cls_avg_keys:
        # update each class average key to avg_score
        update_cls_avg_dict(
            cls_avg_dict,
            label=cls_avg_key,
            label_map=None,
            metric_keys=metric_keys,
            metric_scores=avg_scores,
        )

    print("### for binary merged ###")

    # binary segmentation is also automatically considered for multiclass task
    # binary task score is done by binary-thresholding the sitk Image
    gt_bin = sitk.BinaryThreshold(
        gt,
        lowerThreshold=1,
    )
    pred_bin = sitk.BinaryThreshold(
        pred,
        lowerThreshold=1,
    )

    metric_scores = metric_func(gt=gt_bin, pred=pred_bin, label=1)

    # update merged binary class to metric_scores
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        # use the label_map for merged bin instead of "1"
        label=BIN_CLASS_LABEL_MAP["1"],
        label_map=None,
        metric_keys=metric_keys,
        metric_scores=metric_scores,
    )

    print("\ncls_avg_dict =")
    pprint.pprint(cls_avg_dict, sort_dicts=False)
    return cls_avg_dict


def update_cls_avg_dict(
    cls_avg_dict, label, label_map, metric_keys, metric_scores
) -> None:
    """
    under a label or some class-average string,
    for each metric_key in metric_keys,
    updates cls_avg_dict by appending
        {metric_key:metric_score}

    Params
    ------
    cls_avg_dict:
        dict to mutate on

    label:
        label can be int (voxel_label) or str (class averge key)

        when label is f"ClsAvg{metric_key}"
        is for_cls_avg, only updates its corresponding metric_key

        when label is voxel_label int
        uses label_map and updates all metric_keys

    label_map:
        MUL_CLASS_LABEL_MAP or BIN_CLASS_LABEL_MAP

    metric_keys:
        list of metric_key for cls_avg_dict

    metric_scores:
        scores that correspond to the metric_keys 1-to-1
    """
    print("\n::update_cls_avg_dict()::\n")
    print(f"label = {label}")
    print(f"metric_keys = {metric_keys}")
    print(f"metric_scores = {metric_scores}")

    # when metric_scores is a single value like Dice
    # ensure the type is list
    if not isinstance(metric_scores, list):
        print("metric_scores NOT a list, convert to list")
        metric_scores = [metric_scores]

    # make sure metric_keys and metric_scores have same dim
    assert len(metric_keys) == len(metric_scores), "metric keys != len(scores)!"

    for_cls_avg = False

    if isinstance(label, str):
        # string label text is its own label
        label_value = label

        # string label text can be
        # for cls_avg_key OR merged_bin

        if label_value.startswith("ClsAvg"):
            # for cls_avg_key str
            for_cls_avg = True
            print("is for_cls_avg")
    else:
        # for voxel_label int
        # label is acquired from label_map
        label_value = label_map[str(label)]

    # initialize a {"label": label_value} for updating
    cls_avg_dict[str(label)] = {
        "label": label_value,
    }

    # update the cls_avg_dict for each metric_key:metric_score
    for index, metric_key in enumerate(metric_keys):
        # when label is cls_avg_key string,
        # only update its corresponding metric_key
        if for_cls_avg and (label_value != f"ClsAvg{metric_key}"):
            print(f"Skip {metric_key}")
            continue
        cls_avg_dict[str(label)][metric_key] = metric_scores[index]

    print("\n::update_cls_avg_dict() DONE! ::\n")


def update_metrics_dict(
    *, cls_avg_dict: dict, metrics_dict: dict, key: str, metric_name: str
):
    """
    to update the metrics_dict in evaluation.py:
    concat the metric_name with the label's value
    assign metric score to the concat_name in metrics_dict

    key and metric_name must exist in the cls_avg_dict
        - cls_avg_dict[key]["label"]
        - cls_avg_dict[key][metric_name]

    example update for label-22 and label-42:
    cls_avg_dict = {
        "22": {"label": "Catch", "Metric1": 314},
        "42": {"label": "dolphin", "Metric1": 2024},
    }
    metrics_dict == {
        "Metric1_Catch": 314,
        "Metric1_dolphin": 2024,
    }
    """
    concat_name = f"{metric_name}_" + cls_avg_dict[key]["label"]
    metrics_dict[concat_name] = cls_avg_dict[key][metric_name]
