import logging

from src.prompt_map import PromptMapper

logger = logging.getLogger(__name__)


def preprocess_dataset(
        task,
        dataset,
        tokenizer,
        prompt,
        num_proc=4,
        batch_size=1,
        use_only_correct_choice: bool = False
):
    prompt_mapper = PromptMapper(
        prompt,
        tokenizer,
        num_proc=num_proc,
        batch_size=batch_size
    )
    result = prompt_mapper(task, dataset)

    def tok(prompt_str, output_str, choices, idx):
        labels_tokenized = tokenizer(output_str, max_length=256, truncation=True)
        if use_only_correct_choice:
            choices_tokenized = labels_tokenized['input_ids']
        else:
            choices_tokenized = tokenizer(choices, max_length=256, truncation=True)['input_ids']
            choices_tokenized = [i for choice_ids in choices_tokenized for i in choice_ids]

        out = {
            "labels"           : labels_tokenized['input_ids'],
            "labels_len"       : len(labels_tokenized['input_ids']),
            "idx"              : idx,
            "choices_tokenized": choices_tokenized,
            **tokenizer(prompt_str, max_length=1024, truncation=True)
        }
        out["input_len"] = len(out['input_ids'])
        return out

    logger.info(f"Tokenizing the dataset")
    tokenized = result.map(
        tok,
        input_columns=["prompt", "output", "choices"],
        remove_columns=result.column_names,
        num_proc=num_proc,
        with_indices=True

    ).sort('input_len', reverse=True)

    return tokenized, result, prompt
