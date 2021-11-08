import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from unittest.mock import MagicMock
import torch
from promptsource.templates import DatasetTemplates
import json

from src import preprocessing


def test_preprocess_dataset():
    ds = load_dataset("anli", split="train_r1[:16]")
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    prompt_task = 'anli'
    prompt_name = 'can we infer'

    task_prompt_templates = DatasetTemplates(prompt_task)
    prompt = task_prompt_templates[prompt_name]
    result, original, used_prompt = preprocessing.preprocess_dataset(
        "test",
        ds,
        tokenizer,
        prompt=prompt,
        batch_size=1,
        num_proc=1
    )

    assert used_prompt == prompt

    def tok(ex, idx):
        prompt_str, target_str = prompt.apply(ex)
        labels_tokenized = tokenizer(target_str, max_length=256, truncation=True)
        out = {
            "idx"       : idx,
            "labels"    : labels_tokenized['input_ids'],
            "labels_len": len(labels_tokenized['input_ids']),
            **tokenizer(prompt_str, max_length=1024, truncation=True)
        }
        out["input_len"] = len(out['input_ids'])
        return out

    expected_ds = ds.map(  # type:ignore
        tok,
        remove_columns=ds.column_names,
        with_indices=True
    ).sort('input_len', reverse=True)

    for i, expected in enumerate(expected_ds):
        assert result[i]['labels_len'] == expected['labels_len']
        assert result[i]['labels'] == expected['labels']
        assert result[i]['input_ids'] == expected['input_ids']
        assert result[i]['attention_mask'] == expected['attention_mask']
