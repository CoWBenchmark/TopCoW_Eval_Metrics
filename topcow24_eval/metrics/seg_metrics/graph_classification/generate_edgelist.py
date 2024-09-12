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

import numpy as np
from topcow24_eval.constants import MUL_CLASS_LABEL_MAP
from topcow24_eval.utils.utils_mask import extract_labels, get_label_by_name

from .edge_criteria import has_A1, has_P1


def generate_edgelist(mask_arr: np.array) -> list[list[int], list[int]]:
    """
    generate edge list of two (1,4) vectors

    input:
        mask_arr: mask np.array

    Returns
        [anterior, posterior] of list[list[int], list[int]]
    """
    print("\ngenerate_edgelist()...\n")

    # get the unique labels
    unique_labels = extract_labels(mask_arr)

    # get the label integers for Acom, 3rd-A2, Pcom
    label_Acom = get_label_by_name("Acom", MUL_CLASS_LABEL_MAP)
    label_trd_A2 = get_label_by_name("3rd-A2", MUL_CLASS_LABEL_MAP)
    label_L_Pcom = get_label_by_name("L-Pcom", MUL_CLASS_LABEL_MAP)
    label_R_Pcom = get_label_by_name("R-Pcom", MUL_CLASS_LABEL_MAP)

    # get the label integers for ACAs
    label_L_ACA = get_label_by_name("L-ACA", MUL_CLASS_LABEL_MAP)
    label_R_ACA = get_label_by_name("R-ACA", MUL_CLASS_LABEL_MAP)

    # get the label integers for PCAs
    label_L_PCA = get_label_by_name("L-PCA", MUL_CLASS_LABEL_MAP)
    label_R_PCA = get_label_by_name("R-PCA", MUL_CLASS_LABEL_MAP)

    # anterior
    print("anterior edgelist...")
    L_A1 = label_L_ACA in unique_labels and has_A1(mask_arr, "L")
    Acom = label_Acom in unique_labels
    trd_A2 = label_trd_A2 in unique_labels
    R_A1 = label_R_ACA in unique_labels and has_A1(mask_arr, "R")

    # posterior
    print("posterior edgelist...")
    L_Pcom = label_L_Pcom in unique_labels
    L_P1 = label_L_PCA in unique_labels and has_P1(mask_arr, "L")
    R_P1 = label_R_PCA in unique_labels and has_P1(mask_arr, "R")
    R_Pcom = label_R_Pcom in unique_labels

    ant_list = [int(edge) for edge in (L_A1, Acom, trd_A2, R_A1)]
    pos_list = [int(edge) for edge in (L_Pcom, L_P1, R_P1, R_Pcom)]

    edgelist = [ant_list, pos_list]

    print(f"\ngenerate_edgelist() [ant_list, pos_list] => {edgelist}")
    return edgelist
