import pandas as pd
from aggregate_all_graph_dicts import aggregate_all_graph_dicts


def test_aggregate_all_graph_dicts_2classes():
    """
    2 classes for anterior and posterior
    BUT in total there are 4 classes in both y_true and y_pred
    we only considers the classes in y_true

    ant_y_true:  ['Var_Ant_0_1_0_1', 'Var_Ant_1_1_0_1', 'Var_Ant_0_1_0_1']
    ant_y_pred:  ['Var_Ant_0_1_1_0', 'Var_Ant_1_1_1_1', 'Var_Ant_0_1_0_1']

    for anterior, there are two classes:
        Var_Ant_0_1_0_1 -> recall 0.5
        Var_Ant_1_1_0_1 -> recall 0
    and two NaN classes:
        Var_Ant_0_1_1_0 -> recall NaN
        Var_Ant_1_1_1_1 -> recall NaN

    pos_y_true:  ['Var_Pos_1_1_1_1', 'Var_Pos_1_0_1_0', 'Var_Pos_1_1_1_1']
    pos_y_pred:  ['Var_Pos_1_0_1_1', 'Var_Pos_0_0_0_0', 'Var_Pos_1_1_1_1']

    for posterior, there are also two classes:
    Var_Pos_1_1_1_1 -> acc 0.5
    Var_Pos_1_0_1_0 -> acc 0
    """
    # List of dictionaries
    dicts = [
        {
            "anterior": {"gt_graph": [0, 1, 0, 1], "pred_graph": [0, 1, 1, 0]},
            "posterior": {"gt_graph": [1, 1, 1, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [1, 1, 0, 1], "pred_graph": [1, 1, 1, 1]},
            "posterior": {"gt_graph": [1, 0, 1, 0], "pred_graph": [0, 0, 0, 0]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 0, 1], "pred_graph": [0, 1, 0, 1]},
            "posterior": {"gt_graph": [1, 1, 1, 1], "pred_graph": [1, 1, 1, 1]},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_graph_dicts = pd.Series(dicts)

    var_bal_acc_dict = aggregate_all_graph_dicts(all_graph_dicts)

    assert var_bal_acc_dict == {"anterior": 0.25, "posterior": 0.25}


def test_aggregate_all_graph_dicts_3classes():
    """
    3 classes for anterior and posterior but highly imbalanced

    ant_y_true:  ['Var_Ant_0_0_0_1', 'Var_Ant_0_1_1_0', 'Var_Ant_0_1_0_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_0_0_1']
    ant_y_pred:  ['Var_Ant_0_1_1_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_0_1_0', 'Var_Ant_0_1_0_0', 'Var_Ant_0_1_1_0', 'Var_Ant_0_0_0_1']

    Var_Ant_0_0_0_1: acc = 1/2
    Var_Ant_0_1_1_0: acc = 2/4
    Var_Ant_0_1_0_0: acc = 0/1

    ant_bal_acc = (0.5 + 0.5 + 0) / 3 = 1/3

    pos_y_true:  ['Var_Pos_1_0_0_0', 'Var_Pos_1_0_0_1', 'Var_Pos_1_0_1_1', 'Var_Pos_1_0_0_1', 'Var_Pos_1_0_0_1', 'Var_Pos_1_0_0_1', 'Var_Pos_1_0_0_0']
    pos_y_pred:  ['Var_Pos_1_0_0_0', 'Var_Pos_1_0_1_1', 'Var_Pos_1_0_1_1', 'Var_Pos_1_0_1_1', 'Var_Pos_1_0_1_1', 'Var_Pos_0_0_0_0', 'Var_Pos_1_0_1_1']

    Var_Pos_1_0_0_0: acc = 1/2
    Var_Pos_1_0_0_1: acc = 0/4
    Var_Pos_1_0_1_1: acc = 1/1

    pos_bal_acc = (0.5 + 0 + 1) / 3 = 0.5

    """
    # List of dictionaries
    dicts = [
        {
            "anterior": {"gt_graph": [0, 0, 0, 1], "pred_graph": [0, 1, 1, 0]},
            "posterior": {"gt_graph": [1, 0, 0, 0], "pred_graph": [1, 0, 0, 0]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 1, 0], "pred_graph": [0, 1, 1, 0]},
            "posterior": {"gt_graph": [1, 0, 0, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 0, 0], "pred_graph": [0, 1, 1, 0]},
            "posterior": {"gt_graph": [1, 0, 1, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 1, 0], "pred_graph": [0, 0, 1, 0]},
            "posterior": {"gt_graph": [1, 0, 0, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 1, 0], "pred_graph": [0, 1, 0, 0]},
            "posterior": {"gt_graph": [1, 0, 0, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 1, 0], "pred_graph": [0, 1, 1, 0]},
            "posterior": {"gt_graph": [1, 0, 0, 1], "pred_graph": [0, 0, 0, 0]},
        },
        {
            "anterior": {"gt_graph": [0, 0, 0, 1], "pred_graph": [0, 0, 0, 1]},
            "posterior": {"gt_graph": [1, 0, 0, 0], "pred_graph": [1, 0, 1, 1]},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_graph_dicts = pd.Series(dicts)

    var_bal_acc_dict = aggregate_all_graph_dicts(all_graph_dicts)

    assert var_bal_acc_dict == {"anterior": 1 / 3, "posterior": 0.5}


def test_aggregate_all_graph_dicts_5classes():
    """
    5 classes in ascending or descending binary numbers for y_true

    ant_y_true:  ['Var_Ant_0_0_0_0', 'Var_Ant_0_0_0_1', 'Var_Ant_0_0_1_0', 'Var_Ant_0_0_1_1', 'Var_Ant_0_1_0_0']
    ant_y_pred:  ['Var_Ant_0_0_1_0', 'Var_Ant_1_1_1_1', 'Var_Ant_0_1_0_1', 'Var_Ant_0_0_1_1', 'Var_Ant_0_1_0_0']

    Var_Ant_0_0_1_1 and Var_Ant_0_1_0_0 matched! -> 2/5

    pos_y_true:  ['Var_Pos_1_1_1_1', 'Var_Pos_1_1_1_0', 'Var_Pos_1_1_0_1', 'Var_Pos_1_1_0_0', 'Var_Pos_1_0_1_1']
    pos_y_pred:  ['Var_Pos_1_0_1_1', 'Var_Pos_1_1_1_0', 'Var_Pos_1_1_0_1', 'Var_Pos_1_1_1_1', 'Var_Pos_1_1_1_1']

    Var_Pos_1_1_1_0 and Var_Pos_1_1_0_1 matched! -> 2/5
    """
    # List of dictionaries
    dicts = [
        {
            "anterior": {"gt_graph": [0, 0, 0, 0], "pred_graph": [0, 0, 1, 0]},
            "posterior": {"gt_graph": [1, 1, 1, 1], "pred_graph": [1, 0, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 0, 0, 1], "pred_graph": [1, 1, 1, 1]},
            "posterior": {"gt_graph": [1, 1, 1, 0], "pred_graph": [1, 1, 1, 0]},
        },
        {
            "anterior": {"gt_graph": [0, 0, 1, 0], "pred_graph": [0, 1, 0, 1]},
            "posterior": {"gt_graph": [1, 1, 0, 1], "pred_graph": [1, 1, 0, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 0, 1, 1], "pred_graph": [0, 0, 1, 1]},
            "posterior": {"gt_graph": [1, 1, 0, 0], "pred_graph": [1, 1, 1, 1]},
        },
        {
            "anterior": {"gt_graph": [0, 1, 0, 0], "pred_graph": [0, 1, 0, 0]},
            "posterior": {"gt_graph": [1, 0, 1, 1], "pred_graph": [1, 1, 1, 1]},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_graph_dicts = pd.Series(dicts)

    var_bal_acc_dict = aggregate_all_graph_dicts(all_graph_dicts)

    assert var_bal_acc_dict == {"anterior": 2 / 5, "posterior": 2 / 5}
