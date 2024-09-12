"""
generate the edgelist label for cow graph classification task

the edgelist is saved as a list
inside have two (1,4) of vectors for anterior and posterior.

E.g. [[1,1,1,1], [1,1,1,1]] is the following edgelist:

anterior:
    L-A1:    1
    Acom:    1
    3rd-A2:  1
    R-A1:    1
posterior:
    L-Pcom:  1
    L-P1:    1
    R-P1:    1
    R-Pcom:  1

Schema for anterior = L-A1, Acom, 3rd-A2, R-A1
vector of length 4

Schema for posterior = L-Pcom, L-P1, R-P1, R-Pcom
vector of length 4

(do not analyze for A2, P2, or BA)
"""


def edge_dict_to_list(edge_dict: dict) -> list[list[int], list[int]]:
    """
    convert an edge_list to an edge list of two (1,4) vectors
        [anterior, posterior]

    Similar to
        /seg_metrics/graph_classification/generate_edgelist.py

    Input:
        edge_dict: dictionary from parse_edge_{json|yml}
        e.g.:
        {
            "anterior": {
                "L-A1": 1,
                "Acom": 0,
                "3rd-A2": 0,
                "R-A1": 1,
            },
            "posterior": {
                "L-Pcom": 1,
                "L-P1": 0,
                "R-P1": 0,
                "R-Pcom": 1,
            },
        }
    Returns:
        edge_list of two lists
        [anterior, posterior] ->
        [[1,0,0,1], [1,0,0,1]]
    """
    # anterior
    anterior = edge_dict["anterior"]
    L_A1 = anterior["L-A1"]
    Acom = anterior["Acom"]
    trd_A2 = anterior["3rd-A2"]
    R_A1 = anterior["R-A1"]

    # posterior
    posterior = edge_dict["posterior"]
    L_Pcom = posterior["L-Pcom"]
    L_P1 = posterior["L-P1"]
    R_P1 = posterior["R-P1"]
    R_Pcom = posterior["R-Pcom"]

    ant_list = [int(edge) for edge in (L_A1, Acom, trd_A2, R_A1)]
    pos_list = [int(edge) for edge in (L_Pcom, L_P1, R_P1, R_Pcom)]

    edgelist = [ant_list, pos_list]

    print(f"\nedge_dict_to_list() => {edgelist}")
    return edgelist
