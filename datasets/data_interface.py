import importlib
import inspect

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataInterface(pl.LightningDataModule):
    """Lightning data module for dynamically loading the configured dataset."""

    def __init__(
        self,
        train_batch_size=64,
        train_num_workers=8,
        test_batch_size=1,
        test_num_workers=1,
        dataset_name=None,
        **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self._load_dataset_class()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.train_dataset = self._instantiate_dataset(state="train")
            self.val_dataset = self._instantiate_dataset(state="val")
        if stage in ("test", None):
            self.test_dataset = self._instantiate_dataset(state="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.train_num_workers,
            pin_memory=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.test_num_workers,
            pin_memory=False,
            shuffle=False,
        )

    def _load_dataset_class(self):
        if not self.dataset_name:
            raise ValueError("dataset_name must be provided.")
        try:
            module = importlib.import_module(f"datasets.{self.dataset_name}")
            self.dataset_class = getattr(module, self.dataset_name)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Invalid dataset module or class: {self.dataset_name}"
            ) from exc

    def _instantiate_dataset(self, **overrides):
        parameters = inspect.signature(self.dataset_class.__init__).parameters
        arguments = {
            key: value
            for key, value in self.kwargs.items()
            if key in parameters and key != "self"
        }
        arguments.update(overrides)
        return self.dataset_class(**arguments)
