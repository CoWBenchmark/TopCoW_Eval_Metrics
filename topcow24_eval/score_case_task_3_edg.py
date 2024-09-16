import os

from topcow24_eval.metrics.edg_metrics.graph_dict_from_files import (
    graph_dict_from_files,
)


def score_case_task_3_edg(
    *, gt_path: str | os.PathLike, pred_path: str | os.PathLike, metrics_dict: dict
) -> None:
    """
    score_case() for Task-3-CoW-Classification

    # task-3 uses metrics/edg_metrics/graph_dict_from_files.py

    work with Path of gt_path pred_path and mutate the metrics_dict object
    """
    graph_dict = graph_dict_from_files(
        gt_edg_path=gt_path,
        pred_edg_path=pred_path,
    )

    # the graph_dict from graph_dict_from_files()
    # is the same as the graph_dict seg_metrics.graph_classification
    # except it has an extra field "distance"

    # extract the graph_dict excluding the 'distance' field
    graph_dict_wo_dist, anterior_distance, posterior_distance = filter_out_distance(
        graph_dict
    )

    # add the common parts of the graph_dict
    # under the column `all_graph_dicts`
    # similar to score_case_task_1_seg
    metrics_dict["all_graph_dicts"] = graph_dict_wo_dist

    # add the distances as separate columns for later score_aggregates
    metrics_dict["anterior_distance"] = anterior_distance
    metrics_dict["posterior_distance"] = posterior_distance


def filter_out_distance(graph_dict: dict) -> tuple[dict, float, float]:
    """filter out the distance field from graph_dict"""
    graph_dict_wo_dist = {
        key: {k: v for k, v in value.items() if k != "distance"}
        for key, value in graph_dict.items()
    }

    anterior_distance = graph_dict["anterior"]["distance"]
    posterior_distance = graph_dict["posterior"]["distance"]

    return graph_dict_wo_dist, anterior_distance, posterior_distance
