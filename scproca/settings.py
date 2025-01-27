import os
import torch
import logging
import random
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

from typing import Optional, Union

logger = logging.getLogger("scProca")


class Settings:
    def __init__(
            self,
            seed: Optional[int] = None,
            device_num: Optional[int] = None,
            batch_size: int = None,
            verbosity: int = logging.INFO,
    ):
        self.seed = seed
        self.device = device_num
        self.verbosity = verbosity
        self.batch_size = batch_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_num: Union[int, None] = None):
        if device_num is None:
            self._device = torch.device("cpu")
        else:
            self._device = torch.device(f"cuda:{device_num}")
            logger.info(f"Using cuda {device_num} device")

    @property
    def seed(self) -> int:
        """Random seed for torch and numpy."""
        return self._seed

    @seed.setter
    def seed(self, seed: Union[int, None] = None):
        """Random seed for torch and numpy."""
        if seed is None:
            self._seed = None
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

            os.environ['PYTHONHASHSEED'] = str(seed)
            self._seed = seed
            logger.info(f"Setting seed {seed}")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        if batch_size is None:
            self._batch_size = None
        else:
            self._batch_size = batch_size
            logger.info(f"Setting batch size {batch_size}")

    @property
    def verbosity(self) -> int:
        """Verbosity level (default `logging.INFO`)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, level: Union[str, int]):
        self._verbosity = level
        logger.setLevel(level)
        if len(logger.handlers) == 0:
            console = Console(force_terminal=True)
            if console.is_jupyter is True:
                console.is_jupyter = False
            ch = RichHandler(level=level, show_path=False, console=console, show_time=False)
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        else:
            logger.setLevel(level)


settings = Settings()
