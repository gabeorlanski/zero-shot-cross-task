import argparse
import logging
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Model
from datasets import load_dataset
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from src.common import prepare_environment
from src.evaluation import evaluate_dataset_with_prompt
import re

FILE_NAME_CLEANER = re.compile(r'[_\.\- ]')


def run(args):
    """
    Run an experiment

    Args:
        args: CLI Args
    """

    # Setup the seeds and logging
    seed = 1966
    numpy_seed = 1991
    torch_seed = 1999
    try:
        prompt_task, prompt_name = args.task_prompt.split("|")
    except ValueError:
        prompt_task = None
        prompt_name = args.task_prompt

    # Cleaning the experiment name to make it savable and readable (somewhat)
    experiment_name = f"{FILE_NAME_CLEANER.sub('_', args.task).replace('/', '_')}" \
                      f"{'.' + FILE_NAME_CLEANER.sub('_', args.subset) if args.subset else ''}"
    prompt_file_name = f"{FILE_NAME_CLEANER.sub('_', args.split)}"
    if prompt_task is not None:
        prompt_file_name += f":{FILE_NAME_CLEANER.sub('_', prompt_task).replace('/', '_')}" \
                            f".{FILE_NAME_CLEANER.sub('_', prompt_name).replace('/', '_')}"
    else:
        prompt_file_name += f":{FILE_NAME_CLEANER.sub('_', prompt_name).replace('/', '_')}"

    out_path, results_path, logger = prepare_environment(
        experiment_name,
        prompt_file_name,
        args.force,
        seed=seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed
    )
    logger.info(f"Starting experiment with name '{experiment_name}")
    logger.info(f"Loading task {args.task} with model {args.model_name}")

    # Load the correct dataset for this task.
    if args.subset:
        dataset = load_dataset(args.task, args.subset, split=args.split)
        prompt_task = f"{args.task}/{args.subset}"
    else:
        dataset = load_dataset(args.task, split=args.split)
        prompt_task = args.task

    evaluate_dataset_with_prompt(
        experiment_name=experiment_name,
        task=args.task,
        dataset=dataset,
        prompt_task=prompt_task,
        prompt_name=prompt_name,
        model_name=args.model_name,
        results_path=results_path,
        batch_size=args.batch_size,
        use_base_model=args.base_model,
        num_beams=args.beams
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="Name of the task")
    parser.add_argument("split", type=str, help="split name")
    parser.add_argument("task_prompt", type=str, help="Name of the Task|Name of Prompt")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        required=False,
        help="overwrite the output directory if it exists",
    )
    parser.add_argument(
        "--model-name",
        "-model",
        type=str,
        default="t5-base",
        help="Model to use",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Subset of the dataset to use",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=8,
        help="batch size",
    )

    parser.add_argument(
        "--beams",
        type=int,
        default=1,
        help="Number of beams",
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        default=False,
        help="Use the base model instead of T5ForConditionalGeneration.",
    )
    run(parser.parse_args())
