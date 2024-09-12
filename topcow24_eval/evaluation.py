"""
The most important file is evaluation.py.
This is the file where you will extend the Evaluation class
and implement the evaluation for your challenge

inherits BaseEvaluation's .evaluate()
"""

from enum import Enum
from os import PathLike
from typing import Optional

import numpy as np
import SimpleITK as sitk
from pandas import DataFrame

from topcow24_eval.base_algorithm import MySegmentationEvaluation
from topcow24_eval.metrics.seg_metrics.clDice import clDice
from topcow24_eval.metrics.seg_metrics.cls_avg_b0 import betti_number_error_all_classes
from topcow24_eval.metrics.seg_metrics.cls_avg_dice import dice_coefficient_all_classes
from topcow24_eval.metrics.seg_metrics.cls_avg_hd95 import hd95_all_classes
from topcow24_eval.metrics.seg_metrics.detection_grp2_labels import (
    detection_grp2_labels,
)
from topcow24_eval.metrics.seg_metrics.generate_cls_avg_dict import update_metrics_dict
from topcow24_eval.metrics.seg_metrics.graph_classification import graph_classification
from topcow24_eval.metrics.seg_metrics.topology_matching import topology_matching
from topcow24_eval.utils.utils_box import get_dcm_slice, parse_roi_txt
from topcow24_eval.utils.utils_nii_mha_sitk import access_sitk_attr


class TopCoWEvaluation(MySegmentationEvaluation):
    def __init__(
        self,
        track: Enum,
        task: Enum,
        expected_num_cases: int,
        need_crop: bool,
        predictions_path: Optional[PathLike] = None,
        ground_truth_path: Optional[PathLike] = None,
        output_path: Optional[PathLike] = None,
        roi_path: Optional[PathLike] = None,
    ):
        super().__init__(
            track,
            task,
            expected_num_cases,
            need_crop,
            predictions_path,
            ground_truth_path,
            output_path,
            roi_path,
        )

    def score_case(self, *, idx: int, case: DataFrame) -> dict:
        """
        inherits from evalutils BaseEvaluation class

        Loads gt&pred images, checks them, extracts ROI size and location,
        crops gt&pred, and return metrics.
        """
        print(f"\n-- call score_case(idx={idx})")
        print(f"case =\n{case.to_dict()}")
        gt_path = case["path_ground_truth"]  # from merge() suffixes
        pred_path = case["path_prediction"]  # from merge() suffixes

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Check that they're the right images
        if (
            self._file_loader.hash_image(gt) != case["hash_ground_truth"]
            or self._file_loader.hash_image(pred) != case["hash_prediction"]
        ):
            raise RuntimeError("Images do not match")

        print("\n>>> before cropping")
        print("gt original attr:")
        access_sitk_attr(gt)
        print("pred original attr:")
        access_sitk_attr(pred)

        if self.need_crop:
            roi_txt_path = case["path_roi_txt"]  # from _load_roi_cases()

            # Get ROI slice
            size_arr, location_arr = parse_roi_txt(roi_txt_path)
            roi_slice = get_dcm_slice(gt.GetSize(), size_arr, location_arr)

            # Crop SimpleITK.Image with slice directly!
            gt = gt[roi_slice]
            pred = pred[roi_slice]

            print("\n<<< after cropping")
            print("gt cropped attr:")
            access_sitk_attr(gt)
            print("pred cropped attr:")
            access_sitk_attr(pred)
        else:
            print("\n<<< No cropping!")

        # Cast to the same type
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(sitk.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        # make sure they have the same metadata
        # Copies the Origin, Spacing, and Direction from the gt image
        # NOTE: metadata like image.GetPixelIDValue() and image.GetPixelIDTypeAsString()
        # are NOT copied from source image
        pred.CopyInformation(gt)

        # Get arrays. Reordering axis from (z,y,x) to (x,y,z)
        gt_arr = sitk.GetArrayFromImage(gt).transpose((2, 1, 0)).astype(np.uint8)
        pred_arr = sitk.GetArrayFromImage(pred).transpose((2, 1, 0)).astype(np.uint8)

        # Score the case

        metrics_dict = {}

        # (1) add Dice for each class
        dice_dict = dice_coefficient_all_classes(gt=gt, pred=pred, task=self.task)
        for key in dice_dict:
            update_metrics_dict(
                cls_avg_dict=dice_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="Dice",
            )

        # (2) add clDice
        cl_dice = clDice(v_p_pred=pred_arr, v_l_gt=gt_arr)
        metrics_dict["clDice"] = cl_dice

        # (3) add Betti0 number error for each class
        betti_num_err_dict = betti_number_error_all_classes(
            gt=gt, pred=pred, task=self.task
        )
        for key in betti_num_err_dict:
            # no more Betti_1 for topcow24
            # and we are interested in the class-average B0 error per case
            update_metrics_dict(
                cls_avg_dict=betti_num_err_dict,
                metrics_dict=metrics_dict,
                key=key,
                metric_name="B0err",
            )

        # (4) add HD95 and HD for each class
        hd_dict = hd95_all_classes(gt=gt, pred=pred, task=self.task)
        for key in hd_dict:
            # class-average key is singular for HD or HD95
            if key == "ClsAvgHD":
                # HD
                update_metrics_dict(
                    cls_avg_dict=hd_dict,
                    metrics_dict=metrics_dict,
                    key=key,
                    metric_name="HD",
                )
            elif key == "ClsAvgHD95":
                # HD95
                update_metrics_dict(
                    cls_avg_dict=hd_dict,
                    metrics_dict=metrics_dict,
                    key=key,
                    metric_name="HD95",
                )
            else:
                # both HD and HD95 for individual CoW labels
                # HD
                update_metrics_dict(
                    cls_avg_dict=hd_dict,
                    metrics_dict=metrics_dict,
                    key=key,
                    metric_name="HD",
                )
                # HD95
                update_metrics_dict(
                    cls_avg_dict=hd_dict,
                    metrics_dict=metrics_dict,
                    key=key,
                    metric_name="HD95",
                )

        # (5) add Group 2 CoW detections
        # each case will have its own detection_dict
        # altogether will be a column of detection_dicts
        # thus name the column as `all_detection_dicts`
        detection_dict = detection_grp2_labels(gt=gt, pred=pred)
        metrics_dict["all_detection_dicts"] = detection_dict

        # (6) add graph classification
        # under the column `all_graph_dicts`
        graph_dict = graph_classification(gt=gt, pred=pred)
        metrics_dict["all_graph_dicts"] = graph_dict

        # (7) add topology matching
        # under the column `all_topo_dicts`
        topo_dict = topology_matching(gt=gt, pred=pred)
        metrics_dict["all_topo_dicts"] = topo_dict

        # add file names
        metrics_dict["pred_fname"] = pred_path.name
        metrics_dict["gt_fname"] = gt_path.name

        return metrics_dict


if __name__ == "__main__":
    from topcow24_eval.configs import expected_num_cases, need_crop, task, track

    evalRun = TopCoWEvaluation(track, task, expected_num_cases, need_crop=need_crop)

    evalRun.evaluate()

    cowsay_msg = """\n
  ____________________________________
< TopCoWEvaluation().evaluate()  Done! >
  ------------------------------------
         \   ^__^ 
          \  (oo)\_______
             (__)\       )\/\\
                 ||----w |
                 ||     ||
    
    """
    print(cowsay_msg)
