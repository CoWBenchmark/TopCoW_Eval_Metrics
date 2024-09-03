"""
Centerline Dice (clDice) on merged binary mask

Metrics for Task-1-CoW-Segmentation
"""

import numpy as np
from skimage.morphology import skeletonize, skeletonize_3d
from topcow24_eval.utils.utils_mask import convert_multiclass_to_binary


def cl_score(*, s_skeleton: np.array, v_image: np.array) -> float:
    """[this function computes the skeleton volume overlap]
    Args:
        s ([bool]): [skeleton]
        v ([bool]): [image]
    Returns:
        [float]: [computed skeleton volume intersection]

    meanings of v, s refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """
    if np.sum(s_skeleton) == 0:
        return 0
    return float(np.sum(s_skeleton * v_image) / np.sum(s_skeleton))


def clDice(*, v_p_pred: np.array, v_l_gt: np.array) -> float:
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]

    meanings of v_l, v_p, s_l, s_p refer to clDice paper:
    https://arxiv.org/abs/2003.07311
    """

    # NOTE: skeletonization works on binary images;
    # need to convert multiclass to binary mask first
    pred_mask = convert_multiclass_to_binary(v_p_pred)
    gt_mask = convert_multiclass_to_binary(v_l_gt)

    # clDice makes use of the skimage skeletonize method
    # see https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html#skeletonize
    if len(pred_mask.shape) == 2:
        call_skeletonize = skeletonize
    elif len(pred_mask.shape) == 3:
        call_skeletonize = skeletonize_3d

    # tprec: Topology Precision
    tprec = cl_score(s_skeleton=call_skeletonize(pred_mask), v_image=gt_mask)
    # tsens: Topology Sensitivity
    tsens = cl_score(s_skeleton=call_skeletonize(gt_mask), v_image=pred_mask)

    if (tprec + tsens) == 0:
        return 0

    return 2 * tprec * tsens / (tprec + tsens)
