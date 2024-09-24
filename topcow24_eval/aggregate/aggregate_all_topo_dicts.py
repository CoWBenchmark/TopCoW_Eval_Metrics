"""
aggregate the topo_dict
from pandas DataFrame: self._case_results
Get the Series: self._case_results["all_topo_dicts"]
topo_dict is under the column `all_topo_dicts`

Variant-balanced topology match rate
"""

import pprint

from pandas import Series
from sklearn.metrics import balanced_accuracy_score
from topcow24_eval.aggregate.edge_list_to_variant_str import edge_list_to_variant_str


def aggregate_all_topo_dicts(all_topo_dicts: Series) -> dict:
    """
    all_topo_dicts is a pandas.Series:
    0 {'gt_topology': {'anterior': {'graph': [0, 0, ...
    1 {'gt_topology': {'anterior': {'graph': [0, 0, ...
    ...

    each topo_dict in all_topo_dicts has the following schema:
        {'gt_topology': {'anterior': {}, 'posterior'{}}
        {'pred_topology': {'anterior': {}, 'posterior'{}}
        'match_verdict': {'anterior': True, 'posterior': True}}

    Similar to aggregate_all_graph_dicts.py,
    we use sklearn.metrics.balanced_accuracy_score
    which considers only the classes in y_true

    when constructing the y_true: use the gt_topology["graph"]
    NOTE: for y_pred, if the match_verdict is True, reuse gt's graph
    but if the match_verdict is False, use `FAIL` as class!
    This is because topology matching can fail even
    if the graph-class may be correct! Thus we differentiate them.

    return dict of variant balanced accuracy for anterior and posterior:
        {"anterior": 0.25, "posterior": 0.25}
    """
    print("\n[aggregate] aggregate_all_topo_dicts()\n")

    # return a dict of two balanced_accuracy_score dicts
    var_bal_acc_dict = {}

    # list of anterior y_true|y_pred for later sklearn.metrics
    list_ant_y_true = []
    list_ant_y_pred = []

    # list of posterior y_true|y_pred for later sklearn.metrics
    list_pos_y_true = []
    list_pos_y_pred = []

    # string for not-matched class
    not_matched = "FAIL"

    for topo_dict in all_topo_dicts.values:
        # use gt_topology's topo graph for y_true and y_pred (if matched)
        gt_ant_topo = topo_dict["gt_topology"]["anterior"]
        gt_pos_topo = topo_dict["gt_topology"]["posterior"]

        # y_true
        ant_y_true = edge_list_to_variant_str(gt_ant_topo["graph"], region="Ant")
        pos_y_true = edge_list_to_variant_str(gt_pos_topo["graph"], region="Pos")

        # add to list of y_true
        list_ant_y_true.append(ant_y_true)
        list_pos_y_true.append(pos_y_true)

        # now for y_pred, first find out if the match_verdict is T/F
        if topo_dict["match_verdict"]["anterior"]:
            # if the match_verdict is True, reuse the gt's graph label
            list_ant_y_pred.append(ant_y_true)
        else:
            # if not matched, append a `FAIL` as label
            list_ant_y_pred.append(not_matched)

        # similarly for posterior
        if topo_dict["match_verdict"]["posterior"]:
            # if the match_verdict is True, reuse the gt's graph label
            list_pos_y_pred.append(pos_y_true)
        else:
            # if not matched, append a `FAIL` as label
            list_pos_y_pred.append(not_matched)

    # the number of y_true, y_pred MUST match
    # the number of cases in all_topo_dicts
    assert (
        all_topo_dicts.size
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

    print("\naggregate_all_topo_dicts =>")
    pprint.pprint(var_bal_acc_dict, sort_dicts=False)

    return var_bal_acc_dict
