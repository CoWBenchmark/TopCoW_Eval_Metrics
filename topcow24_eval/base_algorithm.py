import json
import logging
import os
import pprint
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Optional

from evalutils import ClassificationEvaluation
from evalutils.exceptions import FileLoaderError
from evalutils.io import SimpleITKLoader
from evalutils.validators import NumberOfCasesValidator
from pandas import DataFrame, concat, merge

from topcow24_eval.aggregate.aggregate_all_detection_dicts import (
    aggregate_all_detection_dicts,
)
from topcow24_eval.aggregate.aggregate_all_graph_dicts import aggregate_all_graph_dicts
from topcow24_eval.aggregate.aggregate_all_topo_dicts import aggregate_all_topo_dicts
from topcow24_eval.constants import TASK
from topcow24_eval.utils.tree_view_dir import DisplayablePath

logger = logging.getLogger(__name__)


class MySegmentationEvaluation(ClassificationEvaluation):
    """
    A special case of a classification task
    Submission and ground truth are image files (eg, ITK images)
    Same number images in the ground truth dataset as there are in each submission.
    By default, the results per case are also reported.
    """

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
        self.track = track
        self.task = task
        # sanity switch off if task is not Task-1-CoW-Segmentation
        if self.task != TASK.MULTICLASS_SEGMENTATION:
            self.need_crop = False
        else:
            self.need_crop = need_crop
        self.execute_in_docker = _is_docker()

        print(f"[init] track = {self.track.value}")
        print(f"[init] task = {self.task.value}")
        print(f"[init] need_crop = {self.need_crop}")
        print(f"[init] execute_in_docker = {self.execute_in_docker}")

        if self.execute_in_docker:
            predictions_path = Path("/input/")
            ground_truth_path = Path("/opt/app/ground-truth/")
            output_file = Path("/output/metrics.json")
            roi_path = Path("/opt/app/roi-metadata/") if self.need_crop else None
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

        # mkdir for *_path if not exist
        for _path in (predictions_path, ground_truth_path, output_path):
            Path(_path).mkdir(parents=True, exist_ok=True)

        if self.need_crop:
            Path(roi_path).mkdir(parents=True, exist_ok=True)

        print(f"[path] predictions_path={predictions_path}")
        print(f"[path] ground_truth_path={ground_truth_path}")
        print(f"[path] output_file={output_file}")
        if self.need_crop:
            print(f"[path] roi_path={roi_path}")

        # do not proceed if input|predictions|ground-truth folders are empty
        input_pred_files = [
            name
            for name in os.listdir(predictions_path)
            if os.path.isfile(os.path.join(predictions_path, name))
        ]
        num_input_pred = len(input_pred_files)

        ground_truth_files = [
            name
            for name in os.listdir(ground_truth_path)
            if os.path.isfile(os.path.join(ground_truth_path, name))
        ]
        num_ground_truth = len(ground_truth_files)

        if need_crop:
            roi_txt_files = [
                name
                for name in os.listdir(roi_path)
                if os.path.isfile(os.path.join(roi_path, name))
            ]
            num_roi_txt = len(roi_txt_files)

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

        super().__init__(
            ground_truth_path=ground_truth_path,
            predictions_path=predictions_path,
            # use Default: `None` (alphanumerical) for file_sorter_key
            file_sorter_key=None,
            # NOTE: SimpleITKLoader is subclass of evalutils ImageLoader
            # ImageLoader returns [{"hash": self.hash_image(img), "path": fname}]
            file_loader=SimpleITKLoader(),
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

        self.SLUG_INPUT = f"head-{self.track.value}-angiography"
        if self.task == TASK.MULTICLASS_SEGMENTATION:
            self.SLUG_OUTPUT = "circle-of-willis-multiclass-segmentation"
        elif self.task == TASK.OBJECT_DETECTION:
            self.SLUG_OUTPUT = "circle-of-willis-roi"
        elif self.task == TASK.GRAPH_CLASSIFICATION:
            self.SLUG_OUTPUT = "circle-of-willis-classification"
        else:
            raise ValueError("Unknown task!")

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
            self.mapping_dict = self.load_predictions_json(
                Path("/input/predictions.json")
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
        print(f"self._ground_truth_cases =\n{self._ground_truth_cases}")
        if self.need_crop:
            print(f"self._roi_cases =\n{self._roi_cases}")
        print(f"self._predictions_cases =\n{self._predictions_cases}")

    def _load_cases(self, *, folder: Path) -> DataFrame:
        """
        Overwrite from evalutils.py
        """
        print(f"\n-- call _load_cases(folder={folder})")
        cases = None

        # ground truth rglob for .nii.gz, prediction rglob for .mha
        img_files = list(folder.rglob("*.nii.gz")) + list(folder.rglob("*.mha"))
        for f in sorted(img_files, key=self._file_sorter_key):
            try:
                # class ImageLoader load() returns
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

    def load_predictions_json(self, fname: Path):
        """
        from
        https://grand-challenge.org/documentation/automated-evaluation/

        loads the JSON, loops through the inputs and outputs,
        and then finds the exact filenames for the outputs
        """
        print(f"\n-- call load_predictions_json(fname={fname})")
        cases = {}

        with open(fname, "r") as f:
            entries = json.load(f)

        if isinstance(entries, float):
            raise TypeError(f"entries of type float for file: {fname}")

        for e in entries:
            # Find case name through input file name
            inputs = e["inputs"]

            print("\nself.SLUG_INPUT = ", self.SLUG_INPUT)
            print("len(inputs) = ", len(inputs))
            print("inputs[0] = ", inputs[0])

            name = None
            for input in inputs:
                if input["interface"]["slug"] == self.SLUG_INPUT:
                    name = str(input["image"]["name"])
                    print("*** name = ", name)
                    break  # expecting only a single input
            if name is None:
                raise ValueError(f"No filename found for entry: {e}")

            # Find output value for this case
            outputs = e["outputs"]

            print("\nself.SLUG_OUTPUT = ", self.SLUG_OUTPUT)
            print("len(outputs) = ", len(outputs))
            print("outputs[0] = ", outputs[0])

            for output in outputs:
                if output["interface"]["slug"] == self.SLUG_OUTPUT:
                    pk = output["image"]["pk"]
                    print("*** pk = ", pk)
                    if ".mha" not in pk:
                        pk += ".mha"
                    cases[pk] = name

        return cases

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
            print(f"after 1st merge()=, merged=\n{merged.to_dict()}")

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

        if self.task == TASK.MULTICLASS_SEGMENTATION:
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


def _is_docker():
    """
    check if process.py is run in a docker env
        bash test.sh vs python3 process.py

    from https://stackoverflow.com/questions/43878953/how-does-one-detect-if-one-is-running-within-a-docker-container-within-python
    """
    cgroup = Path("/proc/self/cgroup")
    exec_in_docker = (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and "docker" in cgroup.read_text()
    )
    print(f"exec_in_docker? {exec_in_docker}")
    return exec_in_docker
