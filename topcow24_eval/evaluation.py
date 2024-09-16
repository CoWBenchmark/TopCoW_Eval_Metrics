"""
The most important file is evaluation.py.
This is the file where you will extend the Evaluation class
and implement the evaluation for your challenge

inherits BaseEvaluation's .evaluate()
"""

from os import PathLike
from typing import Optional

from pandas import DataFrame

from topcow24_eval.base_algorithm import MySegmentationEvaluation
from topcow24_eval.constants import TASK, TRACK
from topcow24_eval.score_case_task_1_seg import score_case_task_1_seg
from topcow24_eval.score_case_task_2_box import score_case_task_2_box
from topcow24_eval.score_case_task_3_edg import score_case_task_3_edg
from topcow24_eval.utils.crop_gt_and_pred import crop_gt_and_pred
from topcow24_eval.utils.utils_nii_mha_sitk import access_sitk_attr


class TopCoWEvaluation(MySegmentationEvaluation):
    def __init__(
        self,
        track: TRACK,
        task: TASK,
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

        Loads gt&pred images/files, checks them,
            if need_crop, extracts ROI size and location,
                crops gt&pred,
        Send the gt-pred pair to separate
        score_case_task_{1,2,3}.py functions to compute the metrics
        return metrics.json
        """
        print(f"\n-- call score_case(idx={idx})")
        print(f"case =\n{case.to_dict()}")
        gt_path = case["path_ground_truth"]  # from merge() suffixes
        pred_path = case["path_prediction"]  # from merge() suffixes

        # init an empty metrics.json for scocre_case_task* to populate
        metrics_dict = {}

        if self.task is TASK.MULTICLASS_SEGMENTATION:
            # Load the images for this case
            # segmentation task uses SimpleITKLoader of ImageLoader
            # which has methods .load_image() and .hash_image()
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
                gt, pred = crop_gt_and_pred(roi_txt_path, gt, pred)
            else:
                print("\n<<< No cropping!")

            # mutate the metrics_dict by score_case_task_1_seg()
            score_case_task_1_seg(gt=gt, pred=pred, metrics_dict=metrics_dict)
        else:
            # task-2 and task-3 uses GenericLoader
            # -> [{"hash": self.hash_file(data), "path": fname}]
            # the files are txt, yml, json files
            #
            # load the binary contents and check the hash
            # similar to task-1 above.
            gt = self._file_loader.load_file(gt_path)
            pred = self._file_loader.load_file(pred_path)

            # Check that they're the right images
            if (
                self._file_loader.hash_file(gt) != case["hash_ground_truth"]
                or self._file_loader.hash_file(pred) != case["hash_prediction"]
            ):
                raise RuntimeError("Files do not match")

            if self.task is TASK.OBJECT_DETECTION:
                # mutate the metrics_dict by score_case_task_2_box()
                score_case_task_2_box(
                    gt_path=gt_path, pred_path=pred_path, metrics_dict=metrics_dict
                )
            else:
                # for Task-3-CoW-Classification
                # mutate the metrics_dict by score_case_task_3_edg()
                score_case_task_3_edg(
                    gt_path=gt_path, pred_path=pred_path, metrics_dict=metrics_dict
                )

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
