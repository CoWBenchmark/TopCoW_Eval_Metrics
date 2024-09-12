from edge_dict_to_list import edge_dict_to_list


def test_edge_dict_to_list_usual_order():
    edge_dict = {
        "anterior": {
            "L-A1": 1,
            "Acom": 0,
            "3rd-A2": 1,
            "R-A1": 1,
        },
        "posterior": {
            "L-Pcom": 1,
            "L-P1": 1,
            "R-P1": 0,
            "R-Pcom": 1,
        },
    }
    edge_list = edge_dict_to_list(edge_dict)

    assert edge_list == [[1, 0, 1, 1], [1, 1, 0, 1]]


def test_edge_dict_to_list_mixedup_order():
    """
    the edge_list have anterior list in front, then posterior list
    and list items also follow a strict order as specified,
    irregardless of the order from the edge_dict
    """
    edge_dict = {
        "posterior": {
            "R-Pcom": 1,
            "L-Pcom": 1,
            "L-P1": 1,
            "R-P1": 0,
        },
        "anterior": {
            "3rd-A2": 1,
            "L-A1": 1,
            "R-A1": 1,
            "Acom": 0,
        },
    }
    edge_list = edge_dict_to_list(edge_dict)
    # always fixed order
    assert edge_list == [[1, 0, 1, 1], [1, 1, 0, 1]]

    edge_dict = {
        "posterior": {
            "L-Pcom": 0,
            "R-P1": 0,
            "R-Pcom": 0,
            "L-P1": 1,
        },
        "anterior": {
            "Acom": 0,
            "3rd-A2": 0,
            "L-A1": 1,
            "R-A1": 1,
        },
    }
    edge_list = edge_dict_to_list(edge_dict)
    # always fixed order
    assert edge_list == [[1, 0, 0, 1], [0, 1, 0, 0]]
