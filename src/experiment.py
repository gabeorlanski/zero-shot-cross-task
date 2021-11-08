from typing import Tuple
import logging
from pathlib import Path
import shutil
import torch
import random
import numpy

from datasets import load_dataset

from src.evaluation import evaluate_dataset_with_prompt

logger = logging.getLogger(__name__)


def prepare_environment(
        prompt_name: str,
        force: bool,
        seed,
        numpy_seed,
        torch_seed
) -> Path:
    """
    Prepare environment for the experiment
    Args:
        prompt_name (str): shorthand name for the prompt
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

    # Create the child dir for the results path
    results_path = Path(prompt_name)
    if not results_path.exists():
        results_path.mkdir(parents=True)
    else:
        if not force:
            raise ValueError(f"'{results_path}' exists")

        shutil.rmtree(results_path)
        results_path.mkdir(parents=True)
    return results_path


def experiment(
        cfg,
        prompt,
        dataset_name,
        split,
        tokenizer,
        model,
        experiment_name,
        prompt_file_name,
        seeds,
        subset=None
):
    results_path = prepare_environment(
        prompt_file_name,
        cfg["force"],
        **seeds
    )
    logger.info(f"Starting experiment with name '{experiment_name}")
    logger.info(f"Loading task {cfg['task']} with model {cfg['model_name']}")

    # Load the correct dataset for this task.
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    original_ds = evaluate_dataset_with_prompt(
        task=experiment_name,
        dataset=dataset,
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        results_path=results_path,
        batch_size=cfg['batch_size'],
        num_beams=cfg['beams'],
        force_generation=cfg.get("force_generation", False),
        length_normalization=cfg.get('length_normalization', False),
        num_proc=cfg.get('num_proc', 1)
    )
    return original_ds,results_path
