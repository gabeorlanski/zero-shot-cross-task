import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset, set_caching_enabled
from promptsource.templates import DatasetTemplates

from src import preprocessing


@pytest.mark.parametrize('only_correct', [True, False], ids=["OnlyCorrect", "Normal"])
@pytest.mark.parametrize('lowercase_choices', [True, False], ids=["LowerCase", "NormalCase"])
def test_preprocess_dataset(only_correct, lowercase_choices):
    set_caching_enabled(False)
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
        num_proc=1,
        use_only_correct_choice=only_correct,
        lowercase_choices=lowercase_choices
    )

    assert used_prompt == prompt

    def get_prompt_tok(ex, id):
        prompt_str, target_str = prompt.apply(ex)
        choices = prompt.get_answer_choices_list(ex)

        choices_to_tokenize = choices
        if lowercase_choices:
            choices_to_tokenize = list(map(lambda c: c.lower(), choices))

        choice_tokenized = tokenizer(
            choices_to_tokenize,
            max_length=256,
            truncation=True,
            add_special_tokens=False
        )['input_ids']

        ex['prompt'] = prompt_str
        ex['output'] = target_str
        ex['choices'] = choices
        ex['choice_ids'] = choice_tokenized
        return ex

    def tok(ex, idx):
        choices_tokenized = tokenizer(
            ex['choices'] if not lowercase_choices else list(
                map(lambda c: c.lower(), ex['choices'])),
            max_length=256,
            truncation=True
        )['input_ids']
        choice_tokenized = [x for e in choices_tokenized for x in e]
        labels_tokenized = tokenizer(ex['output'], max_length=256, truncation=True)
        if only_correct:
            choice_tokenized = labels_tokenized['input_ids']

        out = {
            "idx"              : idx,
            "labels"           : labels_tokenized['input_ids'],
            "labels_len"       : len(labels_tokenized['input_ids']),
            "choices_tokenized": choice_tokenized,
            **tokenizer(ex['prompt'], max_length=1024, truncation=True)
        }
        out["input_len"] = len(out['input_ids'])
        return out

    expected_normal_ds = ds.map(
        get_prompt_tok,
        with_indices=True
    )

    for i, expected in enumerate(expected_normal_ds):
        assert set(original[i]) == set(expected)
        for k, v in expected.items():
            assert k in original[i], k
            assert original[i][k] == v, f"result[{i}][{k}]"

    expected_toked_ds = expected_normal_ds.map(  # type:ignore
        tok,
        remove_columns=expected_normal_ds.column_names,
        with_indices=True
    ).sort('idx')
    result = result.sort("idx")

    for i, (actual, expected) in enumerate(zip(result, expected_toked_ds)):
        assert actual['labels_len'] == expected['labels_len']
        assert actual['labels'] == expected['labels']
        assert actual['input_ids'] == expected['input_ids']
        assert actual['attention_mask'] == expected['attention_mask']
        assert actual['choices_tokenized'] == expected['choices_tokenized']
    set_caching_enabled(True)
