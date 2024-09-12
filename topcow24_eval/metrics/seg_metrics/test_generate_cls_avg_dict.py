import SimpleITK as sitk
from generate_cls_avg_dict import (
    generate_cls_avg_dict,
    update_cls_avg_dict,
    update_metrics_dict,
)
from topcow24_eval.constants import TASK


def dummy_metric_func(gt, pred, label: int):
    metric_scores = [42 + label, 2024 + label, 8006 + label]
    return metric_scores


def test_update_cls_avg_dict_for_cls_avg():
    """
    when label is f"ClsAvg{metric_key}"
    for_cls_avg, only updates its corresponding metric_key
    """
    cls_avg_dict = {}

    # label is cls_avg_key for M1
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label="ClsAvgM1",
        label_map={"not relevant": True},
        metric_keys=["M1", "M2"],
        metric_scores=[3.14, 2.99],
    )

    # cls_avg_dict is now updated
    assert cls_avg_dict == {
        "ClsAvgM1": {
            "label": "ClsAvgM1",
            "M1": 3.14,
        },
    }

    cls_avg_dict = {}

    # label is cls_avg_key for M2
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label="ClsAvgM2",
        label_map={"not relevant": True},
        metric_keys=["M1", "M2"],
        metric_scores=[3.14, 2.99],
    )

    # cls_avg_dict is now updated
    assert cls_avg_dict == {
        "ClsAvgM2": {
            "label": "ClsAvgM2",
            "M2": 2.99,
        },
    }

    cls_avg_dict = {}

    # label is cls_avg_key for B0err
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label="ClsAvgB0err",
        label_map={"not relevant": True},
        metric_keys=["B0err"],
        metric_scores=[42],
    )

    # cls_avg_dict is now updated
    assert cls_avg_dict == {
        "ClsAvgB0err": {
            "label": "ClsAvgB0err",
            "B0err": 42,
        },
    }


def test_update_cls_avg_dict_for_merged_bin():
    """
    when label is string but is for merged binary
    the dict should be:
        "MergedBin": {"label": "MergedBin", "dice_score": 0.88},
    """
    cls_avg_dict = {}

    # label is cls_avg_key for M1
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label="BinBin:)",
        label_map={"not relevant": True},
        metric_keys=["M1", "M2"],
        metric_scores=[3.14, 2.99],
    )

    # cls_avg_dict is now updated with all metric_keys
    assert cls_avg_dict == {
        "BinBin:)": {
            "label": "BinBin:)",
            "M1": 3.14,
            "M2": 2.99,
        },
    }


def test_update_cls_avg_dict_voxel_label():
    """
    when label is voxel_label int
    uses label_map and updates all metric_keys
    """
    cls_avg_dict = {}

    # label is voxel_label int
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label=100,
        label_map={"99": "CAT", "100": "DOG"},
        metric_keys=["M1", "M2"],
        metric_scores=[3.14, 2.99],
    )

    # cls_avg_dict update both M1 and M2
    assert cls_avg_dict == {
        "100": {
            "label": "DOG",
            "M1": 3.14,
            "M2": 2.99,
        }
    }


def test_update_cls_avg_dict_voxel_label_singlular_metric():
    """
    when metric_scores is a singular value
    """
    cls_avg_dict = {}

    # label is voxel_label int
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label=42,
        label_map={"42": "Relax", "19": "Pig"},
        metric_keys=["Dice"],
        metric_scores=12.0,
    )

    assert cls_avg_dict == {"42": {"label": "Relax", "Dice": 12}}


def test_update_cls_avg_dict_append_existing():
    """
    if cls_avg_dict contains values for label
    """
    cls_avg_dict = {
        "100": {
            "label": "DOG",
            "M1": 3.14,
            "M2": 2.99,
        }
    }

    # update label 100 again
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label=100,
        label_map={"99": "CAT", "100": "DOG"},
        metric_keys=["M1", "M2"],
        metric_scores=[-3, -2],
    )

    # label 100 is over-written by new metric_scores
    assert cls_avg_dict == {
        "100": {
            "label": "DOG",
            "M1": -3,
            "M2": -2,
        },
    }

    # update with a class average for M2
    update_cls_avg_dict(
        cls_avg_dict=cls_avg_dict,
        label="ClsAvgM2",
        label_map={"99": "CAT", "100": "DOG"},
        metric_keys=["M1", "M2"],
        metric_scores=[108, 109],
    )

    # rest of dict intact, but append a new entry
    assert cls_avg_dict == {
        "100": {"label": "DOG", "M1": -3, "M2": -2},
        "ClsAvgM2": {"label": "ClsAvgM2", "M2": 109},
    }


#################################################################
# e2e for generate_cls_avg_dict tests


def test_generate_cls_avg_dict_blank_multiclass():
    """
    if gt and pred contains no labels and for multiclass task,
    when there are no labels in the images,
    return blank cls_avg_dict with only average and merged_binary of 0
    with MULTICLASS_SEGMENTATION
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    cls_avg_dict = generate_cls_avg_dict(
        gt=image1,
        pred=image2,
        task=TASK.MULTICLASS_SEGMENTATION,
        # dummy_metric_func returns 4 scores
        metric_keys=["M1", "M2", "M3"],
        metric_func=dummy_metric_func,
    )
    assert cls_avg_dict == {
        "ClsAvgM1": {"label": "ClsAvgM1", "M1": 0},
        "ClsAvgM2": {"label": "ClsAvgM2", "M2": 0},
        "ClsAvgM3": {"label": "ClsAvgM3", "M3": 0},
        "MergedBin": {
            "label": "MergedBin",
            "M1": 0,
            "M2": 0,
            "M3": 0,
        },
    }


def test_generate_cls_avg_dict_only1BA_multiclass():
    """
    if gt and pred contains simple labels and for multiclass task,
    compute the metric_scores for
    all present labels and update the cls_avg_dict
    update each class average key to avg_score

    multi-class segmentation is also automatically considered for binary task
    binary task score is done by binary-thresholding the sitk Image

    with MULTICLASS_SEGMENTATION
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # set some regions to a non-zero label
    image1[0, 0, 0] = 1
    image2[2, 2, 2] = 1

    cls_avg_dict = generate_cls_avg_dict(
        gt=image1,
        pred=image2,
        task=TASK.MULTICLASS_SEGMENTATION,
        # dummy_metric_func returns metric_scores
        metric_keys=["M1", "M2", "M3"],
        metric_func=dummy_metric_func,
    )
    # now matches the output from dummy_metric_func
    assert cls_avg_dict == {
        "1": {"label": "BA", "M1": 43, "M2": 2025, "M3": 8007},
        "ClsAvgM1": {"label": "ClsAvgM1", "M1": 43.0},
        "ClsAvgM2": {"label": "ClsAvgM2", "M2": 2025.0},
        "ClsAvgM3": {"label": "ClsAvgM3", "M3": 8007.0},
        "MergedBin": {"label": "MergedBin", "M1": 43, "M2": 2025, "M3": 8007},
    }


def test_generate_cls_avg_dict_label123_multiclass():
    """
    if gt and pred contains simple labels and for multiclass task,
    compute the metric_scores for
    all present labels and update the cls_avg_dict
    update each class average key to avg_score

    multi-class segmentation is also automatically considered for binary task
    binary task score is done by binary-thresholding the sitk Image

    with MULTICLASS_SEGMENTATION
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # set some regions to a non-zero label
    image1[0, 0, 0] = 1
    image2[2, 2, 2] = 2
    image2[0, 1, 2] = 3

    cls_avg_dict = generate_cls_avg_dict(
        gt=image1,
        pred=image2,
        task=TASK.MULTICLASS_SEGMENTATION,
        # dummy_metric_func returns metric_scores
        metric_keys=["M1", "M2", "M3"],
        metric_func=dummy_metric_func,
    )
    # now matches the output from dummy_metric_func
    assert cls_avg_dict == {
        # value += label
        "1": {"label": "BA", "M1": 43, "M2": 2025, "M3": 8007},
        "2": {"label": "R-PCA", "M1": 44, "M2": 2026, "M3": 8008},
        "3": {"label": "L-PCA", "M1": 45, "M2": 2027, "M3": 8009},
        "ClsAvgM1": {"label": "ClsAvgM1", "M1": 44.0},  # average
        "ClsAvgM2": {"label": "ClsAvgM2", "M2": 2026.0},  # average
        "ClsAvgM3": {"label": "ClsAvgM3", "M3": 8008.0},  # average
        "MergedBin": {"label": "MergedBin", "M1": 43, "M2": 2025, "M3": 8007},
    }


def test_generate_cls_avg_dict_label101112HD_multiclass():
    """
    if gt and pred contains simple labels and for multiclass task,
    compute the metric_scores for
    all present labels and update the cls_avg_dict
    update each class average key to avg_score

    multi-class segmentation is also automatically considered for binary task
    binary task score is done by binary-thresholding the sitk Image

    with MULTICLASS_SEGMENTATION
    """
    image1 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    image2 = sitk.Image([3, 3, 3], sitk.sitkUInt8)
    # set some regions to a non-zero label
    image1[0, 0, 0] = 10
    image2[2, 2, 2] = 11
    image2[0, 1, 2] = 12

    cls_avg_dict = generate_cls_avg_dict(
        gt=image1,
        pred=image2,
        task=TASK.MULTICLASS_SEGMENTATION,
        # use a lambda to return metric_scores
        metric_keys=["HD", "HD95"],
        # metric_func returns a List[float]
        metric_func=lambda gt, pred, label: [label / 10, label / 2],
    )
    # now matches the output from dummy_metric_func
    assert cls_avg_dict == {
        # lambda gt, pred, label: [label / 10, label / 2]
        "10": {
            "label": "Acom",
            "HD": 1.0,
            "HD95": 5.0,
        },
        "11": {
            "label": "R-ACA",
            "HD": 1.1,
            "HD95": 5.5,
        },
        "12": {
            "label": "L-ACA",
            "HD": 1.2,
            "HD95": 6.0,
        },
        "ClsAvgHD": {"label": "ClsAvgHD", "HD": 1.0999999999999999},  # avg = 1.1
        "ClsAvgHD95": {"label": "ClsAvgHD95", "HD95": 5.5},  # average = 5.5
        "MergedBin": {"label": "MergedBin", "HD": 0.1, "HD95": 0.5},
    }


def test_update_metrics_dict():
    """test for update_metrics_dict() used in evaluation.py"""
    cls_avg_dict = {
        "22": {"label": "Catch", "Metric1": 314},
        "42": {"label": "dolphin", "Metric1": 2024},
    }

    metrics_dict = {}

    # update metrics_dict with label-22
    update_metrics_dict(
        cls_avg_dict=cls_avg_dict,
        metrics_dict=metrics_dict,
        key="22",
        metric_name="Metric1",
    )
    # check the updated metrics_dict
    assert metrics_dict == {"Metric1_Catch": 314}

    # update metrics_dict again, now with label-42
    update_metrics_dict(
        cls_avg_dict=cls_avg_dict,
        metrics_dict=metrics_dict,
        key="42",
        metric_name="Metric1",
    )
    # check the updated metrics_dict
    assert metrics_dict == {
        "Metric1_Catch": 314,
        "Metric1_dolphin": 2024,
    }
