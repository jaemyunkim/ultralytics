
from ultralytics.models.yolo.detect.train import *
from typing import Dict, List, Tuple, Union


class DetectionTrainerEx(DetectionTrainer):
    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if isinstance(self.data, dict):
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)    
        elif isinstance(self.data, list):
            index = 0
            for i, d in enumerate(self.data):
                if img_path in d[mode]:
                    index = i
                    break
            data = self.data[index]
            return build_yolo_dataset(self.args, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        if isinstance(self.data, dict):
            self.model.nc = self.data["nc"]  # attach number of classes to model
            self.model.names = self.data["names"]  # attach class names to model
        elif isinstance(self.data, list):
            self.model.nc = self.data[0]["nc"]  # attach number of classes to model
            self.model.names = self.data[0]["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """
        Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        if isinstance(self.data, dict):
            model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        elif isinstance(self.data, list):
            data = self.data[0]
            model = DetectionModel(cfg, nc=data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)

        if weights:
            model.load(weights)
        return model


# Monkey patch
DetectionTrainer.build_dataset = DetectionTrainerEx.build_dataset
DetectionTrainer.get_dataloader = DetectionTrainerEx.get_dataloader
DetectionTrainer.set_model_attributes = DetectionTrainerEx.set_model_attributes
DetectionTrainer.get_model = DetectionTrainerEx.get_model
