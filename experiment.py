import argparse
import logging
from transformers import T5ForConditionalGeneration, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from pathlib import Path
import shutil
import torch

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from src.common import prepare_global_logging
from src.prompt_map import PromptMapper


def run(args):
    """
    Run an experiment

    Args:
        args: CLI Args
    """

    # Setup logging etc
    out_path = Path("results", args.run_name)
    if not out_path.exists():
        out_path.mkdir(parents=True)
    else:
        if not args.force:
            raise ValueError(f"'{out_path}' exists")

        shutil.rmtree(out_path)
        out_path.mkdir(parents=True)
    prepare_global_logging(out_path.resolve().absolute(), log_name="experiment")
    logger = logging.getLogger("experiment")
    logger.info(f"Starting experiment with name '{args.run_name}'")

    logger.info(f"Loading task {args.task} with model {args.model_name}")
    dataset = load_dataset(args.task,download_config=args.subset,split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Prompt it
    from promptsource.templates import DatasetTemplates
    try:
        prompt_task, prompt_name = args.task_prompt.split("|")
    except ValueError:
        prompt_task = args.task
        prompt_name = args.task_prompt
    # Get all the AG News prompts
    ag_news_prompts = DatasetTemplates(prompt_task)
    # Select a prompt by name
    prompt = ag_news_prompts[prompt_name]

    prompt_mapper = PromptMapper.by_name("default")(prompt_name, prompt, 4, batch_size=1)
    result = prompt_mapper("craigslist_bargains", dataset)

    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(torch.device(0))


    def tok(b, v):
        output = tokenizer(v, max_length=256, truncation=True, padding="max_length")
        output = {f'target_{k}': v for k, v in output.items()}
        return {**output, **tokenizer(b, max_length=1024, truncation=True)}

    tokenized = result['validation'].map(
        tok,
        input_columns=["prompt", "output"],
        remove_columns=result['validation'].column_names
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8, max_length=1024,
                                       padding='longest')

    data_loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=16,
        collate_fn=collator
    )

    logger.info(f"Starting Generation")
    pred_file = out_path.joinpath('preds.txt').open('w', encoding="utf-8")
    for b in tqdm(data_loader):
        generated = model.generate(
            input_ids=b['input_ids'].to(torch.device(0)),
            attention_mask=b['attention_mask'].to(torch.device(0)),
            max_length=256
        )
        source = tokenizer.batch_decode(b['input_ids'], skip_special_tokens=True)
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        gold = tokenizer.batch_decode(b['target_input_ids'], skip_special_tokens=True)
        for i, pred in enumerate(preds):
            pred_file.write(f"Source:{source[i]}\n")
            pred_file.write(f"Prediction: {pred}\n")
            pred_file.write(f"Gold: {gold[i]}\n")
            pred_file.write("\n\n\n")
    pred_file.close()
    logger.info("Finished applying the prompt.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", type=str, help="Name for the run.")
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
