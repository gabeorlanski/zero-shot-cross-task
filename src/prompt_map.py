from typing import List, Dict, Optional
import logging

from datasets import Dataset
from promptsource.templates import Template, DatasetTemplates
from transformers import PreTrainedTokenizer
import yaml
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy

from src.common import sanitize_name

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_GROUP = "PROMPTSOURCE"


class PromptMapper:
    """
    Class to map a prompt to a dataset
    """

    def __init__(
            self,
            prompt: Template,
            tokenizer: PreTrainedTokenizer,
            num_proc=1,
            remove_columns=None,
            batch_size=None,
    ):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.remove_columns = remove_columns or []
        self.batch_size = batch_size or 1

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError()

    def __call__(self, task: str, dataset: Dataset) -> Dataset:
        logging.info(f"Preprocessing '{task}' with prompt {self.prompt.name}")

        # Try to preprocess the dataset with a subclass implementation, if that
        # does not work just use the original dataset.
        try:
            preprocessed = self._preprocess_dataset(dataset)
        except NotImplementedError:
            logging.debug("Reverting to default behavior for dataset")
            preprocessed = dataset

        # Map the prompt to the dataset and remove columns as needed.
        if self.batch_size > 1:
            return preprocessed.map(
                lambda b: self.apply_prompt_to_batch(self.prompt, self.tokenizer, b),
                num_proc=self.num_proc,
                remove_columns=self.remove_columns,
                batched=True,
                batch_size=self.batch_size
            )

        return preprocessed.map(
            lambda b: self.apply_prompt_to_example(self.prompt, self.tokenizer, b),
            num_proc=self.num_proc,
            remove_columns=self.remove_columns)

    @staticmethod
    def apply_prompt_to_batch(prompt, tokenizer, batch):
        out = {"prompt": [], "output": [], "choices": [], "choice_ids": []}
        example_num = 0
        keys = list(batch.keys())
        while True:
            failed = False
            example = {}
            for k in keys:
                try:
                    example[k] = batch[k][example_num]
                except IndexError:
                    failed = True,
                    break

            if failed: break

            prompt_str, output_str = prompt.apply(example)
            out['prompt'].append(prompt_str)
            out['output'].append(output_str)
            choices = prompt.get_answer_choices_list(example) or []
            out['choice_ids'].append(tokenizer(choices, add_special_tokens=False)['input_ids'])
            out['choices'].append(choices)
            example_num += 1

        return out

    @staticmethod
    def apply_prompt_to_example(prompt, tokenizer, example):
        prompt_str, output_str = prompt.apply(example)
        choices = prompt.get_answer_choices_list(example) or []
        return {
            "prompt"    : prompt_str,
            "output"    : output_str,
            "choices"   : choices,
            "choice_ids": tokenizer(choices, add_special_tokens=False)['input_ids']
        }


def load_prompts(
        prompt_task: str,
        categories: List,
        prompt_filter_kwargs: Optional[Dict] = None
):
    task_prompt_templates = DatasetTemplates(prompt_task)
    prompt_filter_kwargs = prompt_filter_kwargs or {"name_list": [], "choice_list": []}
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")

    prompts_to_use = []
    for prompt in filter_prompts(task_prompt_templates.templates, **prompt_filter_kwargs):
        # Add the group name and the prompt name to the list of prompts to use.
        prompts_to_use.append(
            (DEFAULT_PROMPT_GROUP, prompt, {
                "name": prompt.name, "category": categories[0], "original_task": [prompt_task]
            })
        )

    return prompts_to_use


def filter_prompts(
        prompt_templates: Dict,
        name_list: Optional[List],
        choice_list: Optional[List],
        choices_in_prompt: Optional[bool] = None,
        original_task: Optional[bool] = None
) -> List:
    if not name_list:
        name_list = None
    if not choice_list:
        choice_list = None

    # Thought this would be easier to implement and expand on but I do not
    # think so anymore.
    filter_fn = [
        ("name_list", name_list, lambda p: name_list is None or p.name in name_list),
        ("choice_list", choice_list,
         lambda p: (
                 choice_list is None
                 or (choice_list == ['null'] and p.answer_choices is None)
                 or (p.answer_choices in choice_list))
         ),
        ("choices_in_prompt", choices_in_prompt,
         lambda p: choices_in_prompt is None or p.metadata.choices_in_prompt == choices_in_prompt
         ),
        ("original_task", original_task,
         lambda p: original_task is None or p.metadata.original_task == original_task
         ),
    ]

    # Go through the prompts and filter with each filter function. If there are
    # no more prompts left raise an error.
    out = list(prompt_templates.values())
    for name, filter_value, fn in filter_fn:
        logger.debug(f"Applying filter '{name}' with values {filter_value}")
        out = list(filter(fn, out))
        if len(out) == 0:
            raise ValueError(f"No prompts found for filter named '{name}' in "
                             f"{filter_value}")

        logger.info(f"{len(out)} prompts after filtering with '{name}'={filter_value}")

    return out


def load_general_prompts(
        prompt_dir: Path,
        prompt_cfg: DictConfig,
        category_filter: Optional[List] = None,
        answer_filter: Optional[List] = None,
        prompt_filter_kwargs: Dict = None
) -> List:
    prompt_path = prompt_dir.joinpath(prompt_cfg['file_name'])
    prompt_metadata = prompt_cfg['prompt_metadata']

    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find general prompt file {prompt_path}")

    prompt_templates = yaml.load(prompt_path.open('r'), yaml.Loader)['templates']
    logger.info(f"Found {len(prompt_templates)} prompts at {prompt_path}")

    logger.info("Validating Prompts")
    found_ids = set(prompt_templates.keys())
    expected_ids = set(prompt_metadata.keys())
    if found_ids != expected_ids:
        logger.critical(f"Missing prompts {found_ids - expected_ids} from the prompt file")
        logger.critical(f"Missing prompts {expected_ids - found_ids} from the prompt config")
        raise ValueError(f"Missing {len(found_ids ^ expected_ids)} prompts")
    out = []

    if answer_filter is not None:
        answer_filter = list(map(set, answer_filter))
        logger.info(f"Answer filter to use is {answer_filter}")

    prompt_filter_kwargs = prompt_filter_kwargs or {"name_list": [], "choice_list": []}
    for prompt in filter_prompts(prompt_templates, **prompt_filter_kwargs):
        current_prompt_metadata = prompt_metadata[prompt.id]
        if prompt.answer_choices != current_prompt_metadata['original_choices']:
            raise ValueError(f"{prompt.name} has incorrect original choices")

        if category_filter and current_prompt_metadata['category'] not in category_filter:
            logger.info(f"Skipping '{prompt.name}' as it is not in filtered category")

        for choices in prompt_cfg['possible_answer_choices']:

            # Check
            if (
                    answer_filter is not None
                    and set(l.strip() for l in choices.split("|||")) not in answer_filter
            ):
                continue

            # Deepcopy to avoid saving a mutable object that will be incorrect
            new_prompt = deepcopy(prompt)
            new_prompt.answer_choices = choices
            choices_str = map(lambda c: c.strip(), choices.split("|||"))
            out.append((
                prompt_cfg['short_name'],
                new_prompt,
                {
                    "name": f"{sanitize_name(prompt.name)}.{''.join(choices_str)}",
                    **current_prompt_metadata
                }
            ))

    if not out:
        raise ValueError("No output prompts")

    return out
