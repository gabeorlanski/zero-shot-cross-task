import argparse
import logging
from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding, \
    DataCollatorForSeq2Seq
from datasets import load_dataset, load_metric
from pathlib import Path
import shutil
import torch

# Prompt it
from promptsource.templates import DatasetTemplates
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from src.common import prepare_environment
from src.prompt_map import PromptMapper
from src.evaluation import evaluate


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

    experiment_name = f"{args.task.replace('/', '_')}.{args.split}" \
                      f"{'.' + args.subset if args.subset else ''}"
    if prompt_task is not None:
        experiment_name += f".{prompt_task.replace('/', '_')}"

    experiment_name += f".{prompt_name.replace(' ', '_').replace('/', '_')}"

    out_path, logger = prepare_environment(
        experiment_name,
        args.force,
        seed=seed,
        numpy_seed=numpy_seed,
        torch_seed=torch_seed
    )
    logger.info(f"Starting experiment with name '{experiment_name}")
    logger.info(f"Loading task {args.task} with model {args.model_name}")

    # Load the correct dataset for this task.
    if args.subset:
        dataset = load_dataset(args.task, args.subset)
        prompt_task = f"{args.task}/{args.subset}"
    else:
        dataset = load_dataset(args.task)
        prompt_task = args.task

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    task_prompt_templates = DatasetTemplates(prompt_task)
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")
    # Select a prompt by name
    prompt = task_prompt_templates[prompt_name]

    prompt_mapper = PromptMapper.by_name("default")(prompt_name, prompt, 4, batch_size=1)
    result = prompt_mapper(args.task, dataset)

    def tok(b, v):
        output = tokenizer(v, max_length=128, truncation=True)
        # output = {f'target_{k}': v for k, v in output.items()}
        out = {"labels": output['input_ids'], **tokenizer(b, max_length=1024, truncation=True)}
        out["input_len"] = len(out['input_ids'])
        return out

    tokenized = result[args.split].map(
        tok,
        input_columns=["prompt", "output"],
        remove_columns=result[args.split].column_names
    ).sort('input_len',reverse=True)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4,
        max_length=1024,
        padding='longest',
        label_pad_token_id=tokenizer.pad_token_id
    )
    data_loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=16,
        collate_fn=collator,
        shuffle=False
    )

    result_file = evaluate(
        task=args.task,
        out_path=out_path,
        data_loader=data_loader,
        model_name=args.model_name,
        tokenizer=tokenizer,
        metrics=prompt.metadata.metrics or ["Accuracy"]
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
    run(parser.parse_args())
