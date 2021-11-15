from typing import Tuple
import logging
from pathlib import Path
import shutil
import random
import numpy
from datasets import load_dataset
import json
import torch
from transformers import T5ForConditionalGeneration, T5Model, AutoTokenizer
from unidecode import unidecode
from omegaconf import OmegaConf

from src.evaluation import evaluate_dataset_with_prompt
from src.tracking import create_run_cfg, save_run_to_wandb
from src.common import sanitize_name
from src.prompt_map import DEFAULT_PROMPT_GROUP

logger = logging.getLogger(__name__)


def prepare_environment(
        prompt_name: str,
        force: bool,
        seed=123,
        numpy_seed=456,
        torch_seed=789,
        working_dir=None
) -> Path:
    """
    Prepare environment for the experiment
    Args:
        prompt_name (str): shorthand name for the prompt
        force (bool): Overwrite existing dirs
        seed (int): seed for python random
        numpy_seed (int): seed for numpy
        torch_seed (int): seed for torch
        working_dir (Path): Working dir to use.

    Returns:
        Output path for saving
        Logger

    """
    working_dir = working_dir or Path.cwd()
    # Set the seeds.
    random.seed(seed)
    numpy.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    # Create the child dir for the results path
    results_path = working_dir.joinpath(prompt_name)
    if not results_path.exists():
        results_path.mkdir(parents=True)
    else:
        if not force:
            raise ValueError(f"'{results_path}' exists")

        shutil.rmtree(results_path)
        results_path.mkdir(parents=True)
    return results_path


def single_experiment(
        cfg,
        prompt,
        dataset_name,
        split,
        tokenizer,
        model,
        experiment_name,
        prompt_file_name,
        seeds,
        subset=None,
        working_dir=None
):
    results_path = prepare_environment(
        prompt_name=prompt_file_name,
        force=cfg["force"],
        working_dir=working_dir,
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
    return original_ds, results_path


def run_experiments(
        cfg,
        prompts_to_use,
        dataset_name,
        split_name,
        task_name,
        verbose_name,
        categories,
        seeds
):
    if cfg.get('base_model', False):
        logger.info("Using base model.")
        model_cls = T5Model
    else:
        logger.info("Using Conditional generation model")
        model_cls = T5ForConditionalGeneration

    logger.info(f"Loading model {cfg['model_name']}")
    # Load these here so we do not need to load them every time
    model = model_cls.from_pretrained(cfg['model_name']).to(torch.device(0))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    completed = 0
    for prompt_group, prompt, prompt_dict in prompts_to_use:
        if not prompt.answer_choices:
            if cfg.get("length_normalization", False) or cfg.get("force_generation", False):
                logger.warning(f"Skipping prompt {prompt.name} because it has"
                               f" no choices and choice specific behaviors are"
                               f" enabled.")
                completed += 1
                continue

        split_file_name = sanitize_name(split_name)
        group_name = sanitize_name(verbose_name)

        # Some of the template names have absurd amount of bad characters.
        # So to remove them we need to first unidecode it, then clean it,
        # then remove duplicate characters
        prompt_fn = prompt_dict['name']
        if prompt_group != DEFAULT_PROMPT_GROUP:
            prompt_fn = f"{prompt_group}.{prompt_fn}"

        # Need to also add some special info if it is length normalized or if
        # generation was forced.
        if cfg.get('base_model', False):
            prompt_fn += ".BaseModel"

        if cfg['length_normalization']:
            prompt_fn += ".LenNorm"
            logger.info("Using Length Normalization")
        elif cfg['force_generation']:
            prompt_fn += ".FG"

        experiment_name = f"{group_name}[{split_file_name}]:{prompt_fn}"

        ds_used, results_path = single_experiment(
            cfg=cfg,
            prompt=prompt,
            dataset_name=dataset_name,
            split=split_name,
            tokenizer=tokenizer,
            model=model,
            experiment_name=experiment_name,
            prompt_file_name=prompt_fn,
            seeds=seeds,
            subset=task_name if dataset_name != task_name else None
        )

        with results_path.joinpath('cfg.yaml').open('w') as cfg_file:
            cfg_file.write(OmegaConf.to_yaml(cfg) + '\n')

        prompt_dict['choice_count'] = max(map(len, ds_used['choices']))
        tags, run_cfg = create_run_cfg(
            cfg=cfg,
            prompt_group=prompt_group,
            prompt=prompt,
            prompt_metadata=prompt_dict
        )
        with results_path.joinpath('run.json').open('w') as f:
            json.dump(run_cfg, f, indent=True)

        # If tracking is enabled we save the results to wandb
        if not cfg.get('disable_tracking', False):
            wandb_group_name = f"{verbose_name}[{split_file_name}]"
            logger.info(f"Saving {prompt_fn} to wandb under group {wandb_group_name}")
            save_run_to_wandb(
                run_cfg=run_cfg,
                tags=tags,
                categories=categories,
                group_name=wandb_group_name,
                name=prompt_fn,
                metrics_path=results_path.joinpath("metrics.json"),
                predictions_path=results_path.joinpath("predictions.jsonl")
            )

        completed += 1
        logger.info(f"Finished {completed}/{len(prompts_to_use)} prompts")

    logger.info(f"Finished all prompts and splits.")
