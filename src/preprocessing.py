import logging
from collections import defaultdict
from typing import Tuple, Optional

from src.prompt_map import PromptMapper
from src.preprocessors import FixedChoiceTaskPreprocessor, TaskMode
from t5.data.preprocessors import rank_classification
from datasets import Dataset
import json
import numpy as np
import tensorflow as tf
from functools import partial

logger = logging.getLogger(__name__)


def preprocess_dataset(
        task,
        dataset,
        tokenizer,
        prompt,
        num_proc=4,
        batch_size=1,
        lowercase_choices: bool = False,
        preprocessor: Optional[FixedChoiceTaskPreprocessor] = None
):
    if preprocessor:
        if not hasattr(prompt.metadata, 'task_mode'):
            raise AttributeError(f"Trying to use a preprocessor but prompt "
                                 f"'{prompt.name}' does not have a task mode")

        if not hasattr(prompt.metadata, 'is_mcq'):
            raise AttributeError(f"Trying to use a preprocessor but prompt "
                                 f"'{prompt.name}' does not have an 'is_mcq'.")

        preprocessor.is_mcq = prompt.metadata.is_mcq
        preprocessor.set_mode(TaskMode[prompt.metadata.task_mode])
        preprocessed_dataset = dataset.map(
            preprocessor,
            with_indices=True,
            remove_columns=dataset.column_names
        )
    else:
        preprocessed_dataset = dataset

    prompt_mapper = PromptMapper(
        prompt,
        tokenizer,
        num_proc=num_proc,
        batch_size=batch_size,
        lowercase_choices=lowercase_choices
    )
    result = prompt_mapper(task, preprocessed_dataset)

    logger.info(f"Tokenizing the dataset")

    # Need to convert it to a TF dataset so that I can use T5's rank
    # classification processor.
    tensor_slices = {column: result[column] for column in result.column_names}
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        tensor_slices
    )
    data = rank_classification(
        tf_dataset,
        inputs_fn=lambda ex: tf.fill((len(ex['choices']),), ex["prompt"]),
        targets_fn=lambda ex: ex['choices'],
        is_correct_fn=lambda ex: tf.equal(ex['choices'], ex['output']),
        mode='eval'
    )

    dataset_records = defaultdict(list)
    for d in data.as_numpy_iterator():
        for k, v in d.items():
            dataset_records[k].append(v)

    final_dataset = Dataset.from_dict(dataset_records).map(
        lambda ex: tokenize_dataset(ex, tokenizer),
        num_proc=num_proc,
        remove_columns=list(dataset_records)
    ).sort('input_len', reverse=True)
    return final_dataset, result, prompt


def tokenize_dataset(
        ex,
        tokenizer
):
    inputs_tokenized = tokenizer(ex['inputs'].decode('utf-8'), max_length=1024, truncation=True)
    labels_tokenized = tokenizer(
        ex['targets'].decode('utf-8'),
        max_length=256,
        truncation=True
    )['input_ids']
    target_tokenized = tokenizer(
        ex['targets'].decode('utf-8'),
        max_length=256,
        truncation=True,
        add_special_tokens=False
    )['input_ids']
    return {
        "idx"       : ex['idx'],
        "is_correct": ex['is_correct'],
        "labels"    : labels_tokenized,
        "labels_len": len(labels_tokenized),
        "target"    : target_tokenized,
        "input_len" : len(inputs_tokenized['input_ids']),
        **inputs_tokenized
    }
