import json
import logging
import re

import hydra
import torch
import wandb
from omegaconf import DictConfig
from promptsource.templates import DatasetTemplates
from transformers import T5ForConditionalGeneration, T5Model, AutoTokenizer
from unidecode import unidecode

from src.common import flatten
from src.experiment import experiment

FILE_NAME_CLEANER = re.compile(r'[^\w]')
DUPE_SPECIAL_CHARS = re.compile(r'([_\.\-])[_\.\-]+')

logger = logging.getLogger(__name__)


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

    if cfg['length_normalization'] and cfg['force_generation']:
        raise ValueError("Cannot have both 'length_normalization' and 'force_generation'")

    # Get the task config
    task_cfg = cfg['task']
    task_name = unidecode(task_cfg['name'])
    verbose_name = task_cfg.get('verbose_name', task_name)
    dataset_name = task_cfg.get("parent_dataset", task_name)
    categories = task_cfg['category']
    if isinstance(categories, str):
        categories = [categories]
    else:
        # To remove the annoying ListConfig object.
        categories = list(categories)

    logger.info(f"Starting experiment with task {verbose_name}")

    if task_name != dataset_name:
        prompt_task = unidecode(f"{dataset_name}/{task_name}")
    else:
        prompt_task = task_name

    task_prompt_templates = DatasetTemplates(prompt_task)
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")

    # Make sure there are no duplicate names. If there are I will need
    # to handle them.
    all_prompts = task_prompt_templates.all_template_names
    if len(set(all_prompts)) != len(all_prompts):
        raise ValueError(f"{prompt_task} has duplicate template names")

    # Get the splits and the prompts we will use. We first check if they have
    # been passed through the CLI. This indicates that the user specifically
    # want to use certain splits. Otherwise we use those from the task.
    split = cfg['split']
    prompt_groups = cfg['task'].get("prompts", {})
    prompt_groupt_to_use = cfg.get('prompt_group', "ALL")

    # Add prompts not in a specific group to the list of default groups
    if "DEFAULT" not in prompt_groups:
        prompt_groups["DEFAULT"] = []

    # Make a list of prompts that are in non-default groups.
    prompt_names_in_groups = [p_name for k, v in prompt_groups.items()
                              for p_name in v if k != "DEFAULT"]
    for prompt in task_prompt_templates.all_template_names:
        if prompt not in prompt_names_in_groups:
            prompt_groups["DEFAULT"].append({"name": prompt, "category": categories[0]})

    prompts_to_use = []
    for prompt_group, prompts in prompt_groups.items():

        # Not a prompt in a group we care about
        if prompt_groupt_to_use != "ALL" and prompt_group != prompt_groupt_to_use:
            continue

        # Add the group name and the prompt name to the list of prompts to use.
        prompts_to_use.extend(map(lambda p: (prompt_group, p), prompts))

    logger.info(f"Using Split {split}")
    logger.info(f"Prompts to use are: {prompts_to_use}")

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
    for prompt_group, prompt_dict in prompts_to_use:
        prompt_name = prompt_dict['name']

        # Select a prompt by name
        prompt = task_prompt_templates[prompt_name]

        if not prompt.answer_choices:
            if cfg.get("length_normalization", False) or cfg.get("force_generation", False):
                logger.warning(f"Skipping prompt {prompt.name} because it has"
                               f" no choices and choice specific behaviors are"
                               f" enabled.")
                completed += 1
                continue

        split_file_name = FILE_NAME_CLEANER.sub('_', unidecode(split)).replace('/', '_')
        group_name = FILE_NAME_CLEANER.sub('_', unidecode(verbose_name)).replace('/', '_')

        # Some of the template names have absurd amount of bad characters.
        # So to remove them we need to first unidecode it, then clean it,
        # then remove duplicate characters
        prompt_fn = FILE_NAME_CLEANER.sub('_', unidecode(prompt.name).replace('/', '_'))
        prompt_fn = DUPE_SPECIAL_CHARS.sub(r'\1', prompt_fn)

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

        ds_used, results_path = experiment(
            cfg=cfg,
            prompt=prompt,
            dataset_name=dataset_name,
            split=split,
            tokenizer=tokenizer,
            model=model,
            experiment_name=experiment_name,
            prompt_file_name=prompt_fn,
            seeds=seeds,
            subset=task_name if dataset_name != task_name else None
        )

        choice_count = max(map(len, ds_used['choices']))
        choice_str = prompt.get_fixed_answer_choices_list()
        if choice_str is not None:
            has_fixed_choices = True
            choice_str = ", ".join(sorted(choice_str))
        else:
            has_fixed_choices = False
            choice_str = f"{choice_count} MCQ" if choice_count > 0 else "N/A"
        run_cfg = {
            "choices_in_prompt"   : prompt.metadata.choices_in_prompt or False,
            "choices"             : choice_str,
            "has_fixed_choices"   : has_fixed_choices,
            "original_task"       : prompt.metadata.original_task,
            "has_choices"         : prompt.answer_choices is not None,
            "prompt_task"         : prompt_task,
            "prompt_name"         : prompt_name,
            "choice_count"        : choice_count,
            "original_prompt_name": prompt.name,
            "prompt_id"           : prompt.id,
            "prompt_category"     : prompt_dict['category'],
            "prompt_group"        : prompt_group,
            "base_model"          : cfg.get('base_model', False),
            "length_normalization": cfg.get('length_normalization'),
            "force_generation"    : cfg.get('force_generation'),
            **flatten(dict(cfg), sep='.')
        }
        if not cfg.get('disable_tracking', False):
            metrics_path = results_path.joinpath("metrics.json")
            preds = results_path.joinpath("predictions.jsonl")
            assert metrics_path.exists()
            assert preds.exists()

            tags = []
            if has_fixed_choices:
                tags.append("Fixed Choices")
            elif choice_count > 0:
                tags.append("MCQ")
            else:
                tags.append("Generation")

            tags.append(f"PromptCat:{prompt_dict.get('category', categories[0])}")
            tags.extend(categories)

            wandb_run = wandb.init(
                project="zero-shot-eval",
                job_type="eval",
                entity="gabeorlanski",
                group=f"{verbose_name}[{split_file_name}]", name=prompt_fn,
                tags=tags,
                config=run_cfg
            )
            metrics = json.loads(metrics_path.read_text('utf-8'))
            # for m in metrics.keys():
            #     if isinstance(metrics[m], list):
            #         metrics[m] = wandb.Histogram(metrics[m])
            wandb_run.log(metrics)
            # Use the ID as the artifact name to guarantee no errors.
            artifact = wandb.Artifact(
                unidecode(f"{group_name}.{split_file_name}.{prompt.id}"),
                type="predictions")
            artifact.add_file(str(preds.resolve().absolute()))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()

        completed += 1
        logger.info(f"Finished {completed}/{len(prompts_to_use)} prompts")

    logger.info(f"Finished all prompts and splits.")


if __name__ == '__main__':
    run()
