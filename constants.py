from enum import Enum


class TRACK(Enum):
    MR = "mr"
    CT = "ct"


class TASK(Enum):
    """
    the values of the strings are different from TopCoWAlgorithm
    """

    BINARY_SEGMENTATION = "binary"
    MULTICLASS_SEGMENTATION = "multiclass"


MUL_CLASS_LABEL_MAP = {
    "0": "Background",
    "1": "BA",
    "2": "R-PCA",
    "3": "L-PCA",
    "4": "R-ICA",
    "5": "R-MCA",
    "6": "L-ICA",
    "7": "L-MCA",
    "8": "R-Pcom",
    "9": "L-Pcom",
    "10": "Acom",
    "11": "R-ACA",
    "12": "L-ACA",
    # "13": "dummy",  # placeholder dummy for label 13
    # "14": "dummy2",  # placeholder dummy for label 14
    "15": "3rd-A2",
}

BIN_CLASS_LABEL_MAP = {
    "0": "Background",
    "1": "CoW",
}
