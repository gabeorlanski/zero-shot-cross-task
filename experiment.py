import argparse
import json

from datasets import load_dataset
import torch
import logging
import os
import hydra
from promptsource.templates import DatasetTemplates
import re
import wandb
from omegaconf import DictConfig, OmegaConf

from src.common import prepare_environment
from src.evaluation import evaluate_dataset_with_prompt

FILE_NAME_CLEANER = re.compile(r'[_\.\- ]')

logger = logging.getLogger(__name__)


def experiment(
        cfg,
        prompt,
        dataset_name,
        split,
        experiment_name,
        split_dir_name,
        prompt_file_name,
        seeds,
        subset=None
):
    out_path, results_path = prepare_environment(
        split_dir_name,
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

    evaluate_dataset_with_prompt(
        task=experiment_name,
        dataset=dataset,
        prompt=prompt,
        model_name=cfg['model_name'],
        results_path=results_path,
        batch_size=cfg['batch_size'],
        use_base_model=cfg.get('base_model', False),
        num_beams=cfg['beams'],
        force_generation=cfg['force_generation']
    )
    return results_path


import collections


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


@hydra.main(config_path='configs', config_name="config")
def run(cfg: DictConfig):
    """
    Run an experiment

    Args:
        cfg: Run configs
    """
    logger.info(f"Starting Experiment.")
    # Setup the seeds and logging
    seeds = dict(
        seed=1966,
        numpy_seed=1991,
        torch_seed=1999
    )

    # Get the task config
    task_cfg = cfg['task']
    task_name = task_cfg['name']
    verbose_name = task_cfg.get('verbose_name', task_name)
    dataset_name = task_cfg.get("parent_dataset", task_name)

    logger.info(f"Starting experiment with task {verbose_name}")

    if task_name != dataset_name:
        prompt_task = f"{dataset_name}/{task_name}"
    else:
        prompt_task = task_name

    task_prompt_templates = DatasetTemplates(prompt_task)
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")

    # Get the splits and the prompts we will use. We first check if they have
    # been passed through the CLI. This indicates that the user specifically
    # want to use certain splits. Otherwise we use those from the task.
    splits_to_use = cfg.get("splits", task_cfg['splits'])
    prompts_to_use = cfg.get("prompts", task_prompt_templates.all_template_names)

    logger.info(f"Splits to use are: {splits_to_use}")
    logger.info(f"Prompts to use are: {splits_to_use}")

    for split in splits_to_use:
        for prompt_name in prompts_to_use:
            # Select a prompt by name
            prompt = task_prompt_templates[prompt_name]

            prompt_fn = f"{FILE_NAME_CLEANER.sub('_', prompt_task).replace('/', '_')}" \
                        f".{FILE_NAME_CLEANER.sub('_', prompt_name).replace('/', '_')}"
            split_fn = f"{FILE_NAME_CLEANER.sub('_', split).replace('/', '_')}"

            group_name = f"{FILE_NAME_CLEANER.sub('_', verbose_name).replace('/', '_')}"

            experiment_name = f"{group_name}:{split_fn}:{prompt_fn}"

            results_path = experiment(
                cfg=cfg,
                prompt=prompt,
                dataset_name=dataset_name,
                split=split,
                experiment_name=experiment_name,
                split_dir_name=split_fn,
                prompt_file_name=prompt_fn,
                seeds=seeds,
                subset=task_name if dataset_name != task_name else None
            )
            choices = prompt.get_fixed_answer_choices_list()
            if choices is not None:
                choices = "|".join(choices)
            run_cfg = {
                "choices_in_prompt": prompt.metadata.choices_in_prompt or False,
                "choices"          : choices,
                "original_task"    : prompt.metadata.original_task,
                "has_choices"      : choices is not None,
                **flatten(dict(cfg), sep='.')
            }

            metrics = results_path.joinpath("metrics.json")
            preds = results_path.joinpath("predictions.jsonl")
            assert metrics.exists()
            assert preds.exists()

            run = wandb.init(
                project="zero-shot-eval",
                job_type="eval",
                entity="gabeorlanski",
                group=f"{verbose_name}[{split_fn}]", name=prompt_fn,
                tags=[prompt_task, prompt_name, cfg['model_name']],
                config=run_cfg
            )
            run.log(json.loads(metrics.read_text('utf-8')))
            artifact = wandb.Artifact(f"{group_name}.{split_fn}.{prompt_fn}", type="predictions")
            artifact.add_file(str(preds.resolve().absolute()))
            run.log_artifact(artifact)
            run.finish()

    logger.info(f"Finished all prompts and splits.")


if __name__ == '__main__':
    run()
