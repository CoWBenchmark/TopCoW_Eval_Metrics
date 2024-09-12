from enum import Enum


class TRACK(Enum):
    MR = "mr"
    CT = "ct"


class TASK(Enum):
    """
    the values of the strings are different from TopCoWAlgorithm
    for segmentation metrics only
    """

    # Task-1-CoW-Segmentation
    MULTICLASS_SEGMENTATION = "multiclass_segmentation"
    # Task-2-CoW-ObjDet
    OBJECT_DETECTION = "object_detection"
    # Task-3-CoW-Classification
    GRAPH_CLASSIFICATION = "graph_classification"


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
    "1": "MergedBin",
}

# NOTE: in case of missing values (FP or FN), set the HD95
# to be roughly the maximum distance in ROI = 90 mm
HD95_UPPER_BOUND = 90

# Group 2 CoW components
GROUP2_COW_COMPONENTS_LABELS = (8, 9, 10, 15)

# IoU threshold for detection of Group 2 CoW components
# a lenient threshold is set to tolerate more detections
IOU_THRESHOLD = 0.25


# detection results
class DETECTION(Enum):
    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"


# for CoW graph classification and topology matching
# Anterior CoW components
ANTERIOR_LABELS = (10, 11, 12, 15)
# Posterior CoW components
POSTERIOR_LABELS = (2, 3, 8, 9)


# boundary distance is a fixed ratio of each X,Y,Z size
BOUNDARY_DISTANCE_RATIO = 0.2
# NOTE: for a boundary_distance_ratio of 50%,
# boundary_iou is just standard IoU
MAX_DISTANCE_RATIO = 0.5
