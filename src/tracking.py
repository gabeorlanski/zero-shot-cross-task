import json
from copy import copy
from typing import Dict, Tuple, List

from src.common import flatten
from omegaconf import OmegaConf
from promptsource.templates import Template
import wandb
from scipy.special import softmax
import pandas as pd
import numpy as np
from src.common import sanitize_name


def get_prompt_info_for_wandb(
        prompt_group: str,
        prompt: Template,
        prompt_metadata: Dict,
        is_general_prompt: bool,
        prompt_group_cfg: Dict
) -> Tuple[List, Dict]:
    prompt_cfg = {
        "choices_in_prompt"   : prompt.metadata.choices_in_prompt or False,
        "original_task"       : prompt.metadata.original_task,
        "has_choices"         : prompt.answer_choices is not None,
        "prompt_name"         : prompt_metadata['name'],
        "is_general_prompt"   : is_general_prompt,
        "original_prompt_name": prompt.name,
        "prompt_id"           : prompt.id,
        "prompt_category"     : prompt_metadata.get("category", None),
        "prompt_group"        : prompt_group,
        "prompt_group_long"   : prompt_group_cfg.get('name', "Promptsource Prompts")
    }

    choice_str = prompt.get_fixed_answer_choices_list()
    choice_count = prompt_group_cfg.get('choice_count', prompt_metadata.get('choice_count', 0))

    if choice_str is not None:
        has_fixed_choices = True
        choice_str = ", ".join(sorted(choice_str))
    else:
        has_fixed_choices = False
        choice_str = f"{choice_count} MCQ" if choice_count > 0 else "N/A"

    prompt_cfg["choices"] = choice_str
    prompt_cfg["choice_count"] = choice_count
    prompt_cfg["has_fixed_choices"] = has_fixed_choices

    original_choices = prompt_metadata.get("original_choices", prompt.answer_choices)
    prompt_cfg['uses_original_choices'] = prompt.answer_choices == original_choices
    prompt_cfg['original_choices'] = original_choices
    prompt_cfg['original_task'] = list(prompt_metadata['original_task'])

    tags = [
        f"PromptCat:{prompt_cfg['prompt_category']}",
        prompt_cfg['prompt_group'],
        "Generalized Prompt" if is_general_prompt else "Task Specific Prompt"
    ]
    return tags, prompt_cfg


def create_run_cfg(
        cfg,
        prompt_group,
        prompt,
        prompt_metadata
) -> Tuple[List, Dict]:
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_dict['task']['general_prompts'] = None
    run_cfg = {
        "base_model"          : cfg['evaluation'].get('base_model', False),
        "length_normalization": cfg['evaluation'].get('length_normalization'),
        "force_generation"    : cfg['evaluation'].get('force_generation'),
        **flatten(cfg_dict, sep='.')
    }

    is_general_prompt = cfg.get("use_general_prompts", False)

    if is_general_prompt:
        prompt_group_cfg = cfg['task']['general_prompts']
    else:
        prompt_group_cfg = {}

    tags, prompt_cfg = get_prompt_info_for_wandb(
        prompt_group=prompt_group,
        prompt=prompt,
        prompt_metadata=prompt_metadata,
        is_general_prompt=is_general_prompt,
        prompt_group_cfg=prompt_group_cfg
    )

    run_cfg.update(prompt_cfg)
    return tags, run_cfg


def create_predictions_df(predictions_path):
    records = []
    for line in predictions_path.read_text('utf-8').splitlines(False):
        d = json.loads(line)

        if len(d['prediction']) > 1:
            pred, *others = d['prediction']
        else:
            pred = d['prediction'][0]
            others = []

        choice_logits = d.pop('choice_logits')
        d['choice_count'] = len(choice_logits)

        d['pred_id'] = None
        d['target_id'] = None
        if choice_logits:
            normalized = np.array(list(choice_logits.values()))
            normalized = 1 - normalized / normalized.sum()
            normalized /= normalized.sum()
            for i, v in enumerate(sorted(list(choice_logits.keys()))):
                d[f"choice_{i}"] = v
                d[f"c{i}_logit"] = choice_logits[v] if choice_logits else None
                d[f"c{i}_logit_normalized"] = normalized[i] if len(normalized) > 0 else None
                if v == pred:
                    d['pred_id'] = i
                if v == d['target']:
                    d['target_id'] = i

        d['correct'] = pred == d['target']

        d['prediction'] = pred
        d['other_beams'] = others
        records.append(d)
    return pd.DataFrame.from_records(records).sort_values(by=['id'])


def save_run_to_wandb(
        run_name,
        run_cfg,
        tags,
        categories,
        group_name,
        name,
        metrics_path,
        predictions_path,
        is_debug
):
    assert metrics_path.exists()
    assert predictions_path.exists()

    if run_cfg['has_fixed_choices']:
        tags.append("Fixed Choices")
    elif run_cfg['choice_count'] > 0:
        tags.append("MCQ")
    else:
        tags.append("Generation")

    tags.extend(categories)

    wandb_run = wandb.init(
        project="zero-shot-eval",
        job_type="evaluation" if not is_debug else "debugging",
        entity="gabeorlanski",
        group=group_name,
        name=name,
        tags=tags,
        config=run_cfg
    )
    metrics = json.loads(metrics_path.read_text('utf-8'))
    wandb_run.log(metrics)

    pred_table = wandb.Table(dataframe=create_predictions_df(predictions_path))
    wandb_run.log({"predictions": pred_table})
    artifact = wandb.Artifact(f"{sanitize_name(group_name)}.{run_name}", 'predictions')
    artifact.add_dir(metrics_path.parent)
    wandb_run.log_artifact(artifact)

    wandb_run.finish()
