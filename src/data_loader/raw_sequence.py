import numpy as np
import logging
import os

from .. import utils
from . import DataLoaderBase
from . import DATASET_ROOT_DIR

logger = logging.getLogger(__name__)

class RawDataLoader(DataLoaderBase):
    """
    DataLoader for raw sequence data (.txt)
    x, y, timestamp, polarity
    """

    NAME = "RAW"

    # Override
    def __init__(self, config: dict = {}):
        self._HEIGHT = config["height"]
        self._WIDTH = config["width"]
        root_dir: str = config["root"] if config["root"] else DATASET_ROOT_DIR
        self.root_dir: str = os.path.expanduser(root_dir)
        data_dir: str = config["dataset"] if config["dataset"] else self.NAME

        self.dataset_dir: str = os.path.join(self.root_dir, data_dir)
        logger.info(f"Loading directory in {self.dataset_dir}")

        self.gt_flow_available: bool
        self.auto_undistort: bool

        if utils.check_key_and_bool(config, "load_gt_flow"):
            # Not implemented yet
            return NotImplementedError("Ground truth for raw sequence dataloader is not implemented yet.")
        else:
            self.gt_flow_available = False

        if utils.check_key_and_bool(config, "distort"):
            return NotImplementedError("Undistortion for raw sequence dataloader is not implemented yet.")            
        else:
            logger.info("No undistortion.")
            self.auto_undistort = False

    
    def get_sequence(self, sequence_name: str) -> dict:
        """
        Load raw sequence data from .txt file
        """
        sequence_file = os.path.join(self.dataset_dir, f"{sequence_name}.txt")
        if not utils.check_file_utils(sequence_file):
            raise FileNotFoundError(f"Sequence file {sequence_file} not found.")

        with open(sequence_file, "r") as f:
            lines = f.readlines()

        sequence = []
        for line in lines:
            x, y, timestamp, polarity = line.strip().split(" ")
            sequence.append([int(x), int(y), float(timestamp), int(polarity)])

        return sequence
    

    def set_sequence(self, sequence_name: str) -> None:
        """
        Set sequence data to self.events
        """
        self.events = self.get_sequence(sequence_name)
        self.min_timestamp = self.events[0][2]
        self.max_timestamp = self.events[-1][2]


    def load_event(self, start_index: int, end_index: int) -> np.ndarray:
        """
        Load event data between start_index and end_index
        """
        return np.array(self.events[start_index:end_index])
    

    def load_event_by_timestamp(self, timestamp: float) -> np.ndarray:
        """
        Load event data at timestamp
        """
        np_events = np.array(self.events)
        idx = np.where(np_events[:, 2] == timestamp)
        return np_events[idx]

