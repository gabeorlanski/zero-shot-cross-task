from typing import Dict
import logging

from datasets import Dataset
from promptsource.templates import Template
from transformers import PreTrainedTokenizer

from src.common import Registrable

logger = logging.getLogger(__name__)


class PromptMapper(Registrable):
    """
    Class to map a prompt to a dataset
    """

    def __init__(
            self,
            prompt: Template,
            num_proc=1,
            remove_columns=None,
            batch_size=None
    ):
        self.prompt = prompt
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
                lambda b: self.apply_prompt_to_batch(self.prompt, b),
                num_proc=self.num_proc,
                remove_columns=self.remove_columns,
                batched=True,
                batch_size=self.batch_size
            )

        return preprocessed.map(
            lambda b: self.apply_prompt_to_example(self.prompt, b),
            num_proc=self.num_proc,
            remove_columns=self.remove_columns)

    @staticmethod
    def apply_prompt_to_batch(prompt, batch):
        out = {"prompt": [], "output": []}
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
            example_num += 1

        return out

    @staticmethod
    def apply_prompt_to_example(prompt, example):
        prompt_str, output_str = prompt.apply(example)
        return {"prompt": prompt_str, "output": output_str}


@PromptMapper.register("default")
class DefaultPromptMapper(PromptMapper):
    pass

