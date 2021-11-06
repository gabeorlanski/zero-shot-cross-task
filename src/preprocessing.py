import logging
from promptsource.templates import DatasetTemplates

from src.prompt_map import PromptMapper

logger = logging.getLogger(__name__)


def preprocess_dataset(task, dataset, tokenizer, prompt_task, prompt_name, num_proc=4,
                       batch_size=1):
    task_prompt_templates = DatasetTemplates(prompt_task)
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")
    # Select a prompt by name
    prompt = task_prompt_templates[prompt_name]

    prompt_mapper = PromptMapper.by_name("default")(
        prompt_name,
        prompt,
        num_proc=num_proc,
        batch_size=batch_size
    )
    result = prompt_mapper(task, dataset)

    def tok(b, v):
        labels_tokenized = tokenizer(v, max_length=256, truncation=True)
        out = {
            "labels"    : labels_tokenized['input_ids'],
            "labels_len": len(labels_tokenized['input_ids']),
            **tokenizer(b, max_length=1024, truncation=True)
        }
        out["input_len"] = len(out['input_ids'])
        return out

    logger.info(f"Tokenizing the dataset")
    tokenized = result.map(
        tok,
        input_columns=["prompt", "output"],
        remove_columns=result.column_names,
        num_proc=num_proc
    ).sort('input_len', reverse=True)

    return tokenized, prompt
