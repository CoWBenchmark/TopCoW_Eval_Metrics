import os

from topcow24_eval.metrics.box_metrics.iou_dict_from_files import iou_dict_from_files


def score_case_task_2_box(
    *, gt_path: str | os.PathLike, pred_path: str | os.PathLike, metrics_dict: dict
) -> None:
    """
    score_case() for Task-2-CoW-ObjDet

    task-2 uses box_metrics/iou_dict_from_files.py

    work with Path of gt_path pred_path and mutate the metrics_dict object
    """
    iou_dict = iou_dict_from_files(gt_path, pred_path)

    # merge the iou_dict into metrics_dict
    metrics_dict |= iou_dict
