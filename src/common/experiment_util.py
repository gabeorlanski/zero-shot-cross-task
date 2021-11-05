from typing import Tuple
import logging
from pathlib import Path
import shutil
import torch
import random
import numpy

from src.common.log_util import prepare_global_logging


def prepare_environment(
        experiment_name: str,
        force: bool,
        seed,
        numpy_seed,
        torch_seed
) -> Tuple[Path, logging.Logger]:
    """
    Prepare environment for the experiment
    Args:
        experiment_name (str): Name of the experiment
        force (bool): Overwrite existing dirs
        seed (int): seed for python random
        numpy_seed (int): seed for numpy
        torch_seed (int): seed for torch

    Returns:
        Output path for saving
        Logger

    """
    # Set the seeds.
    random.seed(seed)
    numpy.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    # Setup logging etc
    out_path = Path("results", experiment_name)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    else:
        if not force:
            raise ValueError(f"'{out_path}' exists")

        shutil.rmtree(out_path)
        out_path.mkdir(parents=True)

    prepare_global_logging(out_path.resolve().absolute(), log_name="experiment")
    logger = logging.getLogger("experiment")

    logger.info(f"Environment has been prepared for experiment: {experiment_name}")
    return out_path, logger
