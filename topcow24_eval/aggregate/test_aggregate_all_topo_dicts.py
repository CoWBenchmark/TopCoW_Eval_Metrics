import pandas as pd
from aggregate_all_topo_dicts import aggregate_all_topo_dicts


def test_aggregate_all_topo_dicts():
    """
    reuse the topo_dicts from test_topology_matching.py

    list_ant_y_true:  ['Var_Ant_1_1_1_1', 'Var_Ant_0_0_0_0', 'Var_Ant_0_0_0_0', 'Var_Ant_0_1_0_1', 'Var_Ant_0_1_0_1']
    list_ant_y_pred:  ['FAIL', 'Var_Ant_0_0_0_0', 'FAIL', 'FAIL', 'FAIL']

    in total 3 classes from list_ant_y_true:
        Var_Ant_1_1_1_1, Var_Ant_0_0_0_0, Var_Ant_0_1_0_1
        bal_acc = (0 + 0.5 + 0) / 3 = 0.5/3

    list_pos_y_true:  ['Var_Pos_0_0_0_0', 'Var_Pos_1_0_0_1', 'Var_Pos_1_0_1_1', 'Var_Pos_1_1_1_1', 'Var_Pos_0_1_0_1']
    list_pos_y_pred:  ['Var_Pos_0_0_0_0', 'FAIL', 'FAIL', 'FAIL', 'FAIL']

    in total 5 classes from list_pos_y_true:
        Var_Pos_0_0_0_0, Var_Pos_1_0_0_1, Var_Pos_1_0_1_1, Var_Pos_1_1_1_1, Var_Pos_0_1_0_1
        bal_acc = 1 / 5 = 0.2
    """
    # List of dictionaries
    dicts = [
        # 1 from test_topology_matching_anterior_np_arr()
        {
            "gt_topology": {
                "anterior": {
                    "graph": [1, 1, 1, 1],
                    "Acom": {"b0": 1, "neighbors": [4, 6, 12, 15]},
                    "R-ACA": {"b0": 1, "neighbors": [4, 6, 12]},
                    "L-ACA": {"b0": 2, "neighbors": [6, 10, 11, 15]},
                    "3rd-A2": {"b0": 1, "neighbors": [6, 10, 12]},
                },
                "posterior": {
                    "graph": [0, 0, 0, 0],
                    "R-PCA": {"b0": 0, "neighbors": []},
                    "L-PCA": {"b0": 0, "neighbors": []},
                    "R-Pcom": {"b0": 0, "neighbors": []},
                    "L-Pcom": {"b0": 0, "neighbors": []},
                },
            },
            "pred_topology": {
                "anterior": {
                    "graph": [0, 1, 1, 1],
                    "Acom": {"detection": "TP", "b0": 1, "neighbors": [4, 6, 12, 15]},
                    "R-ACA": {"detection": "TP", "b0": 2, "neighbors": [4, 6]},
                    "L-ACA": {"detection": "FN", "b0": 1, "neighbors": [10, 15]},
                    "3rd-A2": {"detection": "TP", "b0": 1, "neighbors": [6, 10, 12]},
                    "LR_flipped": False,
                },
                "posterior": {
                    "graph": [0, 0, 0, 0],
                    "R-PCA": {"detection": "TN", "b0": 0, "neighbors": []},
                    "L-PCA": {"detection": "TN", "b0": 0, "neighbors": []},
                    "R-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                    "L-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": False,
                },
            },
            "match_verdict": {"anterior": False, "posterior": True},
        },
        # 2 from test_topology_matching_posterior_np_arr()
        {
            "gt_topology": {
                "anterior": {
                    "graph": [0, 0, 0, 0],
                    "Acom": {"b0": 0, "neighbors": []},
                    "R-ACA": {"b0": 0, "neighbors": []},
                    "L-ACA": {"b0": 0, "neighbors": []},
                    "3rd-A2": {"b0": 0, "neighbors": []},
                },
                "posterior": {
                    "graph": [1, 0, 0, 1],
                    "R-PCA": {"b0": 1, "neighbors": [3]},
                    "L-PCA": {"b0": 1, "neighbors": [2]},
                    "R-Pcom": {"b0": 1, "neighbors": [9]},
                    "L-Pcom": {"b0": 1, "neighbors": [8]},
                },
            },
            "pred_topology": {
                "anterior": {
                    "graph": [0, 0, 0, 0],
                    "Acom": {"detection": "TN", "b0": 0, "neighbors": []},
                    "R-ACA": {"detection": "TN", "b0": 0, "neighbors": []},
                    "L-ACA": {"detection": "TN", "b0": 0, "neighbors": []},
                    "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": False,
                },
                "posterior": {
                    "graph": [1, 0, 0, 1],
                    "R-PCA": {"detection": "TP", "b0": 2, "neighbors": [3]},
                    "L-PCA": {"detection": "TP", "b0": 1, "neighbors": [2, 9]},
                    "R-Pcom": {"detection": "FN", "b0": 1, "neighbors": [9]},
                    "L-Pcom": {"detection": "TP", "b0": 1, "neighbors": [3, 8]},
                    "LR_flipped": False,
                },
            },
            "match_verdict": {"anterior": True, "posterior": False},
        },
        # 3 from test_topology_matching_antpos_np_arr()
        {
            "gt_topology": {
                "anterior": {
                    "graph": [0, 0, 0, 0],
                    "Acom": {"b0": 0, "neighbors": []},
                    "R-ACA": {"b0": 1, "neighbors": [6, 7]},
                    "L-ACA": {"b0": 1, "neighbors": [8, 9]},
                    "3rd-A2": {"b0": 0, "neighbors": []},
                },
                "posterior": {
                    "graph": [1, 0, 1, 1],
                    "R-PCA": {"b0": 1, "neighbors": [1, 3, 6, 7]},
                    "L-PCA": {"b0": 1, "neighbors": [2, 6, 7]},
                    "R-Pcom": {"b0": 1, "neighbors": [4, 9, 12]},
                    "L-Pcom": {"b0": 1, "neighbors": [4, 6, 8, 12]},
                },
            },
            "pred_topology": {
                "anterior": {
                    "graph": [0, 1, 1, 0],
                    "Acom": {
                        "detection": "FP",
                        "b0": 1,
                        "neighbors": [5, 6, 7, 9, 11, 15],
                    },
                    "R-ACA": {"detection": "TP", "b0": 1, "neighbors": [6, 7, 10, 15]},
                    "L-ACA": {"detection": "TP", "b0": 1, "neighbors": [8, 9]},
                    "3rd-A2": {"detection": "FP", "b0": 1, "neighbors": [10, 11]},
                    "LR_flipped": False,
                },
                "posterior": {
                    "graph": [1, 0, 1, 1],
                    "R-PCA": {"detection": "TP", "b0": 1, "neighbors": [1, 3, 5, 6, 7]},
                    "L-PCA": {"detection": "TP", "b0": 1, "neighbors": [2, 6, 7]},
                    "R-Pcom": {"detection": "TP", "b0": 1, "neighbors": [4, 5, 9, 12]},
                    "L-Pcom": {
                        "detection": "TP",
                        "b0": 1,
                        "neighbors": [4, 5, 6, 8, 10, 12],
                    },
                    "LR_flipped": False,
                },
            },
            "match_verdict": {"anterior": False, "posterior": False},
        },
        # 4 from test_topology_matching_topcow003ct()
        {
            "gt_topology": {
                "anterior": {
                    "graph": [0, 1, 0, 1],
                    "Acom": {"b0": 1, "neighbors": [11, 12]},
                    "R-ACA": {"b0": 1, "neighbors": [4, 10]},
                    "L-ACA": {"b0": 1, "neighbors": [10]},
                    "3rd-A2": {"b0": 0, "neighbors": []},
                },
                "posterior": {
                    "graph": [1, 1, 1, 1],
                    "R-PCA": {"b0": 1, "neighbors": [1, 8]},
                    "L-PCA": {"b0": 1, "neighbors": [1, 9]},
                    "R-Pcom": {"b0": 1, "neighbors": [2, 4]},
                    "L-Pcom": {"b0": 1, "neighbors": [3, 6]},
                },
            },
            "pred_topology": {
                "anterior": {
                    "graph": [1, 0, 0, 0],
                    "Acom": {"detection": "FN", "b0": 0, "neighbors": []},
                    "R-ACA": {"detection": "FN", "b0": 1, "neighbors": [6, 12]},
                    "L-ACA": {"detection": "FN", "b0": 2, "neighbors": [6, 11]},
                    "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": True,
                },
                "posterior": {
                    "graph": [0, 1, 1, 0],
                    "R-PCA": {"detection": "FN", "b0": 1, "neighbors": [1, 3]},
                    "L-PCA": {"detection": "FN", "b0": 1, "neighbors": [1, 2]},
                    "R-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                    "L-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                    "LR_flipped": False,
                },
            },
            "match_verdict": {"anterior": False, "posterior": False},
        },
        # 5 from test_topology_matching_topcow023mr()
        {
            "gt_topology": {
                "anterior": {
                    "graph": [0, 1, 0, 1],
                    "Acom": {"b0": 1, "neighbors": [11, 12]},
                    "R-ACA": {"b0": 1, "neighbors": [4, 10]},
                    "L-ACA": {"b0": 1, "neighbors": [10]},
                    "3rd-A2": {"b0": 0, "neighbors": []},
                },
                "posterior": {
                    "graph": [0, 1, 0, 1],
                    "R-PCA": {"b0": 1, "neighbors": [8]},
                    "L-PCA": {"b0": 1, "neighbors": [1]},
                    "R-Pcom": {"b0": 1, "neighbors": [2, 4]},
                    "L-Pcom": {"b0": 0, "neighbors": []},
                },
            },
            "pred_topology": {
                "anterior": {
                    "graph": [0, 0, 0, 0],
                    "Acom": {"detection": "FN", "b0": 0, "neighbors": []},
                    "R-ACA": {"detection": "FN", "b0": 0, "neighbors": []},
                    "L-ACA": {"detection": "FN", "b0": 0, "neighbors": []},
                    "3rd-A2": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": True,
                },
                "posterior": {
                    "graph": [0, 1, 1, 0],
                    "R-PCA": {"detection": "FN", "b0": 1, "neighbors": [1]},
                    "L-PCA": {"detection": "FN", "b0": 1, "neighbors": [1]},
                    "R-Pcom": {"detection": "FN", "b0": 0, "neighbors": []},
                    "L-Pcom": {"detection": "TN", "b0": 0, "neighbors": []},
                    "LR_flipped": True,
                },
            },
            "match_verdict": {"anterior": False, "posterior": False},
        },
    ]

    # Create a Pandas Series with dictionaries
    all_topo_dicts = pd.Series(dicts)

    var_bal_acc_dict = aggregate_all_topo_dicts(all_topo_dicts)

    assert var_bal_acc_dict == {"anterior": 0.5 / 3, "posterior": 1 / 5}
