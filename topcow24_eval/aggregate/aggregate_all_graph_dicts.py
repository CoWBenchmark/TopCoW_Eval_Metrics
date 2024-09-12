"""
aggregate the graph_dict
from pandas DataFrame: self._case_results
Get the Series: self._case_results["all_graph_dicts"]
graph_dict is under the column `all_graph_dicts`

to get the variant-balanced graph classification accuracy
"""

import pprint

from pandas import Series
from sklearn.metrics import balanced_accuracy_score
from topcow24_eval.aggregate.edge_list_to_variant_str import edge_list_to_variant_str


def aggregate_all_graph_dicts(all_graph_dicts: Series) -> dict:
    """
    all_graph_dicts is a pandas.Series:
    0 {'anterior': {'gt_graph': [0,0,0,1], 'pred_graph...} 'posterior': {...}
    1 {'anterior': {'gt_graph': [1,1,1,0], 'pred_graph...} 'posterior': {...}
    ...

    The definition of balanced accuracy differs in sklearn.metrics
    balanced_accuracy_score vs classification_report
    NOTE: the balanced_accuracy_score from sklearn.metrics
    considers only the classes in y_true, while classification_report
    considers all classes in both y_true and y_pred.

    For topcow we use sklearn.metrics.balanced_accuracy_score
    as we do not want to deal with NaN recalls when some
    random variants are predicted, which will dilute the balanced accuracy,
    and we want a consistent denominator for all models.
    Although this does not penalize models that predict way-off FP classes.

    return dict of variant balanced accuracy for anterior and posterior:
        {"anterior": 0.25, "posterior": 0.25}

    For more info on sklearn metrics, see:
    https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    print("\n[aggregate] aggregate_all_graph_dicts()\n")

    # return a dict of two balanced_accuracy_score dicts
    var_bal_acc_dict = {}

    # list of anterior y_true|y_pred for later sklearn.metrics
    list_ant_y_true = []
    list_ant_y_pred = []

    # list of posterior y_true|y_pred for later sklearn.metrics
    list_pos_y_true = []
    list_pos_y_pred = []

    for graph_dict in all_graph_dicts.values:
        # each graph_dict has an anterior and a posterior dict
        # add to the anterior and posterior list separately
        list_ant_y_true.append(
            edge_list_to_variant_str(
                graph_dict["anterior"]["gt_graph"],
                region="Ant",
            ),
        )
        list_ant_y_pred.append(
            edge_list_to_variant_str(
                graph_dict["anterior"]["pred_graph"],
                region="Ant",
            )
        )

        list_pos_y_true.append(
            edge_list_to_variant_str(
                graph_dict["posterior"]["gt_graph"],
                region="Pos",
            ),
        )
        list_pos_y_pred.append(
            edge_list_to_variant_str(
                graph_dict["posterior"]["pred_graph"],
                region="Pos",
            )
        )

    # the number of y_true, y_pred MUST match
    # the number of cases in all_graph_dicts
    assert (
        all_graph_dicts.size
        == len(list_ant_y_true)
        == len(list_ant_y_pred)
        == len(list_pos_y_true)
        == len(list_pos_y_pred)
    )

    # log the list for debugging
    print("list_ant_y_true: ", list_ant_y_true)
    print("list_ant_y_pred: ", list_ant_y_pred)
    print("list_pos_y_true: ", list_pos_y_true)
    print("list_pos_y_pred: ", list_pos_y_pred)

    ant_bal_acc = balanced_accuracy_score(list_ant_y_true, list_ant_y_pred)
    print(f"ant_bal_acc = {ant_bal_acc}")

    pos_bal_acc = balanced_accuracy_score(list_pos_y_true, list_pos_y_pred)
    print(f"pos_bal_acc = {pos_bal_acc}")

    var_bal_acc_dict["anterior"] = ant_bal_acc
    var_bal_acc_dict["posterior"] = pos_bal_acc

    print("\naggregate_all_graph_dicts =>")
    pprint.pprint(var_bal_acc_dict, sort_dicts=False)

    return var_bal_acc_dict
