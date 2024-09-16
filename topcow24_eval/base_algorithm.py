import json
import logging
import os
import pprint
from os import PathLike
from pathlib import Path
from typing import Optional

from evalutils import ClassificationEvaluation
from evalutils.exceptions import FileLoaderError
from evalutils.io import FileLoader, SimpleITKLoader
from evalutils.validators import NumberOfCasesValidator
from pandas import DataFrame, concat, merge, set_option

from topcow24_eval.aggregate.aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
)
from topcow24_eval.aggregate.aggregate_all_graph_dicts import aggregate_all_graph_dicts
from topcow24_eval.aggregate.aggregate_all_topo_dicts import aggregate_all_topo_dicts
from topcow24_eval.constants import SLUG_OUTPUT, TASK, TRACK
from topcow24_eval.for_gc_docker import is_docker, load_predictions_json
from topcow24_eval.utils.tree_view_dir import DisplayablePath

logger = logging.getLogger(__name__)

# display more content when printing pandas
set_option("display.max_columns", None)
set_option("max_colwidth", None)


class MySegmentationEvaluation(ClassificationEvaluation):
    """
    A special case of a classification task
    Submission and ground truth are image files (eg, ITK images)
    Same number images in the ground truth dataset as there are in each submission.
    By default, the results per case are also reported.
    """

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
        self.track = track
        self.task = task
        # switch self.need_crop off if task is not Task-1-CoW-Segmentation
        if self.task is not TASK.MULTICLASS_SEGMENTATION:
            self.need_crop = False
        else:
            self.need_crop = need_crop
        self.execute_in_docker = is_docker()

        print(f"[init] track = {self.track.value}")
        print(f"[init] task = {self.task.value}")
        print(f"[init] need_crop = {self.need_crop}")
        print(f"[init] execute_in_docker = {self.execute_in_docker}")

        if self.execute_in_docker:
            predictions_path = Path("/input/")
            ground_truth_path = Path("/opt/app/ground-truth/")
            output_file = Path("/output/metrics.json")
            roi_path = Path("/opt/app/roi-metadata/") if self.need_crop else None
            self.predictions_json = Path("/input/predictions.json")
        else:
            # When not in docker environment, the paths of pred, gt, roi etc
            # are set to be on the same level as package dir `topcow24_eval`
            # Unless they were specified by the user as initialization params

            # Get the path of the current script
            script_path = Path(__file__).resolve()
            print(f"[path] script_path: {script_path}")

            # The resource files (gt, pred, roi etc)
            # are on the same level as package dir `topcow24_eval`
            # thus are two parents of the current script_path
            resource_path = script_path.parent.parent
            print(f"[path] resource_path: {resource_path}")

            predictions_path = (
                Path(predictions_path)
                if predictions_path is not None
                else resource_path / "predictions/"
            )
            ground_truth_path = (
                Path(ground_truth_path)
                if ground_truth_path is not None
                else resource_path / "ground-truth/"
            )
            output_path = (
                Path(output_path)
                if output_path is not None
                else resource_path / "output/"
            )
            output_file = output_path / "metrics.json"
            if self.need_crop:
                roi_path = (
                    Path(roi_path)
                    if roi_path is not None
                    else resource_path / "roi-metadata/"
                )

        # mkdir for out_path if not exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

        print(f"[path] predictions_path={predictions_path}")
        print(f"[path] ground_truth_path={ground_truth_path}")
        print(f"[path] output_file={output_file}")
        if self.need_crop:
            print(f"[path] roi_path={roi_path}")

        # do not proceed if input|predictions|ground-truth folders are empty
        num_input_pred = len(
            [
                str(x)
                for x in predictions_path.rglob("*")
                if x.is_file() and x.name != "predictions.json"
            ]
        )

        num_ground_truth = len(
            [str(x) for x in ground_truth_path.rglob("*") if x.is_file()]
        )

        if self.need_crop:
            num_roi_txt = len(
                [str(x) for x in roi_path.rglob("*") if x.suffix == ".txt"]
            )

        print(f"num_input_pred = {num_input_pred}")
        print(f"num_ground_truth = {num_ground_truth}")

        assert num_input_pred > 0, "no input files"
        assert num_ground_truth > 0, "no ground truth"

        if self.need_crop:
            print(f"num_roi_txt = {num_roi_txt}")
            assert num_roi_txt > 0, "no roi txt"

        for _ in DisplayablePath.make_tree(predictions_path):
            print(_.displayable())
        for _ in DisplayablePath.make_tree(ground_truth_path):
            print(_.displayable())
        if self.need_crop:
            for _ in DisplayablePath.make_tree(roi_path):
                print(_.displayable())

        # slug for input interface (for gc docker)
        self.slug_input = f"head-{self.track.value}-angiography"

        # depending on the task,
        # set slug for outer interface (for gc docker)
        # set file_loader
        # set self.extensions

        if self.task is TASK.MULTICLASS_SEGMENTATION:
            self.slug_output = SLUG_OUTPUT.TASK_1_SEG.value
            # NOTE: SimpleITKLoader is subclass of evalutils ImageLoader
            # ImageLoader returns [{"hash": self.hash_image(img), "path": fname}]
            file_loader = SimpleITKLoader()
            # task-1:
            # ground truth rglob for *.nii.gz, prediction rglob for *.mha
            self.extensions = ("*.nii.gz", "*.nii", "*.mha")
        elif self.task is TASK.OBJECT_DETECTION:
            self.slug_output = SLUG_OUTPUT.TASK_2_BOX.value
            # GenericLoader returns [{"hash": self.hash_file(data), "path": fname}]
            file_loader = GenericLoader()
            # task-2:
            # ground truth rglob for *.txt, prediction rglob for *.json
            self.extensions = ("*.txt", "*.TXT", "*.json", "*.JSON")
        elif self.task is TASK.GRAPH_CLASSIFICATION:
            self.slug_output = SLUG_OUTPUT.TASK_3_EDG.value
            file_loader = GenericLoader()
            # task-3:
            # ground truth rglob for *.yml, prediction rglob for *.json
            self.extensions = ("*.yml", "*.json", "*.JSON")
        else:
            raise ValueError("Unknown task!")

        super().__init__(
            ground_truth_path=ground_truth_path,
            predictions_path=predictions_path,
            # use Default: `None` (alphanumerical) for file_sorter_key
            file_sorter_key=None,
            # file_loader is of type FileLoader from evalutils.io
            # task-1 uses SimpleITKLoader
            # task-2 and task-2 use GenericLoader
            file_loader=file_loader,
            validators=(
                # we use the NumberOfCasesValidator to check that the correct number
                # of cases has been submitted by the challenge participant
                NumberOfCasesValidator(num_cases=expected_num_cases),
                # NOTE: We do not use UniqueIndicesValidator
                # since this might throw an error due to uuid provided by GC
                # NOTE: also do not use UniqueImagesValidator
            ),
            output_file=output_file,
        )

        if self.need_crop:
            self._roi_path = roi_path
            self._roi_cases = DataFrame()
            self._roi_validators = (
                NumberOfCasesValidator(num_cases=expected_num_cases),
            )

        print("Path at terminal when executing this file")
        print(os.getcwd() + "\n")

        print("MySegmentationEvaluation __init__ complete!")

    def load(self):
        """
        three input dataframes
        IMPORTANT: we sort them so the rows match!
        then we merge the dataframes so the correct image is loaded
        """
        print("\n-- call load()")
        self._ground_truth_cases = self._load_cases(folder=self._ground_truth_path)
        self._ground_truth_cases = self._ground_truth_cases.sort_values(
            "path"
        ).reset_index(drop=True)

        if self.need_crop:
            self._roi_cases = self._load_roi_cases(folder=self._roi_path)
            self._roi_cases = self._roi_cases.sort_values("path_roi_txt").reset_index(
                drop=True
            )

        self._predictions_cases = self._load_cases(folder=self._predictions_path)
        # NOTE: how to sort self._predictions_cases depends on if needs predictions.json
        # using mapping_dict to sort predictions according to ground_truth name
        if self.execute_in_docker:
            # from
            # https://grand-challenge.org/documentation/automated-evaluation/

            # the platform also supplies a JSON file that tells you
            # how to map the random output filenames with the
            # original input filenames from the input
            # You as a challenge organizer must, therefore,
            # read /input/predictions.json to map the output filenames
            # with the input filenames
            self.mapping_dict = load_predictions_json(
                fname=self.predictions_json,
                slug_input=self.slug_input,
                slug_output=self.slug_output,
                task=self.task,
            )
            print("******* self.mapping_dict *******")
            pprint.pprint(self.mapping_dict, sort_dicts=False)
            print("****************************")
            # NOTE: predictions.json also used to sort the predictions

            # use the mapping_dict to map the outputs with the
            # actual filenames when computing the metrics in the evaluation script
            # This is done by updating self._predictions_cases["ground_truth_path"]
            # with the contents of mapping_dict
            self._predictions_cases["ground_truth_path"] = [
                self._ground_truth_path / self.mapping_dict[Path(path).name]
                for path in self._predictions_cases.path
            ]
            self._predictions_cases = self._predictions_cases.sort_values(
                "ground_truth_path"
            ).reset_index(drop=True)
        else:
            self._predictions_cases = self._predictions_cases.sort_values(
                "path"
            ).reset_index(drop=True)

        print("*** after sorting ***")
        print(f"\nself._ground_truth_cases =\n{self._ground_truth_cases}")
        if self.need_crop:
            print(f"\nself._roi_cases =\n{self._roi_cases}")
        print(f"\nself._predictions_cases =\n{self._predictions_cases}")

    def _load_cases(self, *, folder: Path) -> DataFrame:
        """
        Overwrite from evalutils.py
        """
        print(f"\n-- call _load_cases(folder={folder})")
        cases = None

        # Use rglob to recursively find all matching extension files,
        # but excluding predictions.json
        files = [
            f
            for ext in self.extensions
            for f in folder.rglob(ext)
            if f.name != "predictions.json"
        ]
        print("rglob files =")
        pprint.pprint(files)

        for f in sorted(files, key=self._file_sorter_key):
            try:
                # class ImageLoader and GenericLoader load() returns
                # [{"hash": self.hash_image(img), "path": fname}]
                new_cases = self._file_loader.load(fname=f)
            except FileLoaderError:
                logger.warning(f"Could not load {f.name} using {self._file_loader}.")
            else:
                if cases is None:
                    cases = new_cases
                else:
                    cases += new_cases

        if cases is None:
            raise FileLoaderError(
                f"Could not load any files in {folder} with {self._file_loader}."
            )

        print("cases = ", cases)
        return DataFrame(cases)

    def _load_roi_cases(self, *, folder: Path) -> DataFrame:
        """
        custom function to load roi cases
        """
        print(f"\n-- call _load_roi_cases(folder={folder})")
        roi_cases = None
        for f in sorted(folder.rglob("*.txt"), key=self._file_sorter_key):
            # type(f) = <class 'pathlib.PosixPath'>
            new_cases = [{"path_roi_txt": f}]
            if roi_cases is None:
                roi_cases = new_cases
            else:
                roi_cases += new_cases

        if roi_cases is None:
            raise FileLoaderError(f"Could not load any files in {folder}")

        print(f"roi_cases = {roi_cases}")
        return DataFrame(roi_cases)

    def validate(self):
        """
        overwrite evalutils
        Validates each dataframe separately
        """
        self._validate_data_frame(df=self._ground_truth_cases)
        self._validate_data_frame(df=self._predictions_cases)
        if self.need_crop:
            self._validate_roi_data_frame(df=self._roi_cases)

    def _validate_roi_data_frame(self, *, df: DataFrame):
        """
        Separate tuple of DataFrameValidators for roi text files
        """
        for validator in self._roi_validators:
            validator.validate(df=df)

    def merge_ground_truth_and_predictions(self):
        """
        overwrite evalutils merge_ground_truth_and_predictions

        Merge gt, preds and roi txt files in one df
        """
        print("\n-- call merge_ground_truth_and_predictions()")
        if self._join_key:
            kwargs = {"on": self._join_key}
        else:
            kwargs = {"left_index": True, "right_index": True}

        assert (
            self._ground_truth_cases.shape[0] == self._predictions_cases.shape[0]
        ), "different number of cases for gt, pred!"

        if self.need_crop:
            assert (
                self._ground_truth_cases.shape[0] == self._roi_cases.shape[0]
            ), "gt and roi numbers do not match!"

            merged = merge(
                left=self._ground_truth_cases,
                right=self._predictions_cases,
                indicator="gt_pred_merge_indicator",
                how="outer",
                # suffixes will only take effect for overlapping column names
                # 1st merge has both named 'path' and 'hash'
                # 2nd merge has different column names, like path_roi_txt
                suffixes=("_ground_truth", "_prediction"),
                **kwargs,
            )
            print("after 1st merge()=, merged=\n")
            pprint.pprint(merged.to_dict(), sort_dicts=False)

            self._cases = merge(
                left=self._roi_cases,
                right=merged,
                indicator=True,
                how="outer",
                **kwargs,
            )
            print(f"after 2nd merge(), self._cases=\n{self._cases}")
        else:
            # NOTE: indicator=True is crucial
            # otherwise cross_validate(self) will complain!
            self._cases = merge(
                left=self._ground_truth_cases,
                right=self._predictions_cases,
                indicator=True,
                how="outer",
                suffixes=("_ground_truth", "_prediction"),
                **kwargs,
            )

        print("\nmerge_ground_truth_and_predictions =>")
        pprint.pprint(self._cases.to_dict(), sort_dicts=False)

    def score(self):
        """
        Overwrite evalutils score()
        from py3.10 evalutils ClassificationEvaluation()
        """
        print("\n-- call score()")

        # NOTE: the NaN in self._case_results DataFrame comes from concat
        # "Columns outside the intersection will be filled with NaN values"

        # self._case_results is a <class 'pandas.core.frame.DataFrame'>
        self._case_results = DataFrame()
        for idx, case in self._cases.iterrows():
            print("\nidx = ", idx)
            print("\ncase = ", case)

            self._case_results = concat(
                [
                    self._case_results,
                    DataFrame.from_records([self.score_case(idx=idx, case=case)]),
                ],
                ignore_index=True,
            )
        # NOTE: self.score_aggregates() -> dict
        # thus self._aggregate_results is a python dictionary
        self._aggregate_results = self.score_aggregates()

        # # Store the DataFrame as a pickle or temporary file to debug it separately
        # self._case_results.to_pickle("self._case_results.pkl")
        # self._case_results.to_csv("self._case_results.csv")

        if self.task is TASK.MULTICLASS_SEGMENTATION:
            # NOTE: for Task-1-CoW-Segmentation
            # work with self._case_results
            # to post-aggregate detection_dict, graph_dict, topo_dict
            # to get the f1-average and variant-average balanced accuracy
            # add the post-aggregate straight to the
            # self._aggregate_results dict

            # metric-5 Average F1 score
            # detection_dict is under the column `all_detection_dicts`
            dect_avg = aggregate_all_detection_dicts(
                self._case_results["all_detection_dicts"]
            )

            # add the dection average dict to self._aggregate_results dict
            self._aggregate_results["dect_avg"] = dect_avg

            # metric-6 Variant-balanced graph classification accuracy
            # graph_dict is under the column `all_graph_dicts`
            graph_var_bal_acc = aggregate_all_graph_dicts(
                self._case_results["all_graph_dicts"]
            )

            # add back to self._aggregate_results dict
            self._aggregate_results["graph_var_bal_acc"] = graph_var_bal_acc

            # metric-7 Variant-balanced topology match rate
            # topo_dict is under the column `all_topo_dicts`
            topo_var_bal_acc = aggregate_all_topo_dicts(
                self._case_results["all_topo_dicts"]
            )

            # add back to self._aggregate_results dict
            self._aggregate_results["topo_var_bal_acc"] = topo_var_bal_acc

        elif self.task is TASK.GRAPH_CLASSIFICATION:
            # NOTE: similarly for Task-3-CoW-Classification
            # work with self._case_results
            # to post-aggregate graph_dict
            # to get variant-average balanced accuracy
            # add the post-aggregate straight to the
            # self._aggregate_results dict

            # metric-1 Variant-balanced graph classification accuracy
            # graph_dict is under the column `all_graph_dicts`
            graph_var_bal_acc = aggregate_all_graph_dicts(
                self._case_results["all_graph_dicts"]
            )

            # add back to self._aggregate_results dict
            self._aggregate_results["graph_var_bal_acc"] = graph_var_bal_acc

    def save(self):
        """
        Overwrite evalutils save()
        from BaseEvaluation()

        add indentation and sorting

        NOTE: self._metrics is a dict with two sub-dicts:
        def _metrics(self) -> Dict:
            {
                "case": self._case_results.to_dict(),
                "aggregates": self._aggregate_results,
            }
        """
        with open(self._output_file, "w") as f:
            f.write(json.dumps(self._metrics, indent=2, sort_keys=True))


############################
# custom file loader for box and edgelist
class GenericLoader(FileLoader):
    """
    A generic file loader for bounding box and edgelist
    compatible with SimpleITKLoader used in task-1

    Since
        box_metrics/iou_dict_from_files.py
        metrics/edg_metrics/graph_dict_from_files.py
    works directly
    with os.PathLike as input, thus our .load() method will
    hash the raw binary of the file and get its path:

    -> [{"hash": self.hash_file(data), "path": fname}]
    """

    def load(self, *, fname: Path) -> list[dict]:
        try:
            data = self.load_file(fname)
        except (ValueError, RuntimeError):
            raise FileLoaderError(
                f"Could not load {fname} using {self.__class__.__qualname__}."
            )
        return [{"hash": self.hash_file(data), "path": fname}]

    @staticmethod
    def load_file(fname: Path) -> bytes:
        return fname.read_bytes()

    @staticmethod
    def hash_file(file_contents: bytes) -> int:
        """similar to ImageLoader's hash_image"""
        return hash(file_contents)
