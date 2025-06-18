from ultralytics.engine.trainer import BaseTrainer
from typing import Dict, List, Tuple, Union
from .utilsEx import (
    check_det_dataset, 
    check_cls_dataset, 
)


class BaseTrainerEx(BaseTrainer):
    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if isinstance(self.args.data, str):
                if self.args.task == "classify":
                    data = check_cls_dataset(self.args.data)
                elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                    "detect",
                    "segment",
                    "pose",
                    "obb",
                }:
                    data = check_det_dataset(self.args.data)
                    if "yaml_file" in data:
                        self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
            elif isinstance(self.args.data, list):
                if self.args.task == "classify":
                    data = []
                    for d in self.args.data:
                        data.append(check_cls_dataset(d))
                elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                    "detect",
                    "segment",
                    "pose",
                    "obb",
                }:
                    data = []
                    out_data = []
                    for d in self.args.data:
                        d = os.path.expandvars(d)   # convert env to real path name
                        data.append(check_det_dataset(d))
                        if "yaml_file" in data[-1]:
                            out_data.append(data[-1]["yaml_file"])  # for validating 'yolo train data=url.zip' usage
                    self.args.data = out_data  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            if isinstance(data, Dict):
                data["names"] = {0: "item"}
                data["nc"] = 1
            elif isinstance(data, list):
                for d in data:
                    d["names"] = {0: "item"}
                    d["nc"] = 1
        return data