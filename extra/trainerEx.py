
from ultralytics.engine.trainer import BaseTrainer
from typing import Dict, List, Tuple, Union

from .utilsEx import (
    check_det_dataset, 
    check_cls_dataset, 
)


class BaseTrainerEx(BaseTrainer):
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp.int(), src=0)  # broadcast from rank 0 to all other ranks; gloo errors with boolean
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        if isinstance(self.data, dict):
            self.train_loader = self.get_dataloader(
                self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
        elif isinstance(self.data, list):
            for d in self.data:
                self.train_loader = self.get_dataloader(
                    d["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
                )
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")


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


# Monkey patch
BaseTrainer._setup_train = BaseTrainerEx._setup_train
BaseTrainer.get_dataset = BaseTrainerEx.get_dataset
