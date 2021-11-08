import logging
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import torch
from t5.evaluation import metrics as mt
import json
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from promptsource.templates import Template
import numpy as np
from functools import lru_cache
from src.preprocessing import preprocess_dataset
from src.common import flatten

logger = logging.getLogger(__name__)

# Dictionary for storing metrics so that we can easily get them from the
# prompt templates
METRICS_DICT = {
    "BLEU"                : mt.bleu,
    "ROUGE"               : mt.rouge,
    "Span Squad"          : mt.span_squad,
    "Squad"               : mt.squad,
    "Trivia QA"           : mt.trivia_qa,
    "Accuracy"            : mt.accuracy,
    "Sequence Accuracy"   : mt.sequence_accuracy,
    "Pearson Correlation" : mt.pearson_corrcoef,
    "Spearman Correlation": mt.spearman_corrcoef,
    "MultiRC"             : mt.multirc_f1_over_all_answers,
    "AUC"                 : mt.auc,
    "COQA F1"             : mt.coqa_f1,
    "Edit Distance"       : mt.edit_distance,
    "Other"               : mt.accuracy,
}


def serialize_prediction(
        prediction,
        target,
        input_seq,
        choice_logits=None
) -> str:
    """
    Helper function to turn predictions into writeable string.

    Args:
        prediction (list): List of prediction strings.
        target (str): The target string.
        input_seq (str): The input string.
        choice_logits (dict): The choices (if exist) dict with log probabilities.

    """
    return json.dumps({
        "prediction"   : prediction,
        "target"       : target,
        "input"        : input_seq,
        "choice_logits": choice_logits or {}
    })


def generate_prediction_sequences(
        out_path,
        data_loader,
        model,
        tokenizer,
        device,
        max_gen_len,
        generator_kwargs: Dict = None
):
    """
    Generate predictions.

    Args:
        out_path (Path): Path to save to.
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
        model: Model to use.
        tokenizer: Tokenizer to use.
        device: device to use.
        max_gen_len (int): Maximum length to generate.
        generator_kwargs: kwargs for generation.

    Returns:
        Path where predictions were saved.
    """
    logger.info(f"Generating Prediction Sequences.")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")
    generator_kwargs = generator_kwargs or {}
    data_iterator = tqdm(data_loader, desc="Generating")
    for batch in data_iterator:
        logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
        generated = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_length=min(max_gen_len, 64),
            early_stopping=True,
            **generator_kwargs
        )

        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        source = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        gold = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        num_beams = generator_kwargs.get('num_return_sequences', 1)
        if num_beams * len(gold) != len(preds):
            raise ValueError(f"Number of beams {num_beams} for {len(gold)} items"
                             f" does not match the number of predictions found {len(preds)}.")

        logger.debug("Saving JSON lines for batch")
        for i, target in enumerate(gold):
            pred_file.write(serialize_prediction(
                preds[i * num_beams:(i + 1) * num_beams],
                target,
                source[i]
            ) + '\n')
    data_iterator.close()
    pred_file.close()
    return pred_path


def generate_predictions_choices(
        out_path,
        data_loader,
        model,
        tokenizer,
        device,
        source_dataset,
        length_normalize=False
):
    """
    Generate predictions when using answer choices.

    Args:
        out_path (Path): Path to save to.
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
        model: Model to use.
        tokenizer: Tokenizer to use.
        device: device to use.
        length_normalize: Use length normalization

    Returns:
        Path where predictions were saved.
    """
    logger.info(f"Generating Choices")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")
    data_iterator = tqdm(data_loader, desc="Generating")

    # Cache in case we are using fixed choices.
    @lru_cache(maxsize=5)
    def tokenize_choice(choice_str):
        return tokenizer(choice_str, add_special_tokens=False)['input_ids']

    with torch.no_grad():
        for batch in data_iterator:
            logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
            input_ids = batch['input_ids'].to(device)
            generated = model(
                input_ids=input_ids,
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            ex_idx = batch['idx']

            source = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            gold = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            choices_by_example = []
            for i in range(len(gold)):

                # Get the choices for this sample
                choices = source_dataset[ex_idx[i].item()]['choices']
                choice_probs = torch.zeros(len(choices))

                # For each choice, sum up the probabilities of it being
                # generated.
                for j, choice in enumerate(choices):
                    choice_ids = tokenize_choice(choice)
                    choice_probs[j] = generated.logits[i, :, choice_ids].sum()
                    if length_normalize:
                        choice_probs[j] /= len(choice_ids)

                choices_by_example.append(
                    {c: p.item() for c, p in zip(choices, choice_probs)}
                )

            logger.debug("Saving JSON lines for batch")
            for i, target in enumerate(gold):
                example_choice_probs = choices_by_example[i]
                prediction_choice = max(
                    example_choice_probs.keys(),
                    key=lambda x: example_choice_probs[x]
                )

                pred_file.write(serialize_prediction(
                    [prediction_choice],
                    target,
                    source[i],
                    example_choice_probs
                ) + '\n')

    data_iterator.close()
    pred_file.close()
    return pred_path


def evaluate(task, prediction_path, metrics, out_path):
    """
    Evaluate a prediction file based on given metrics.
    Args:
        task: Name of the task
        prediction_path: Path to the predictions
        metrics: Metrics to use
        out_path: Where the results can be saved.

    Returns:
        Path to where the metrics.json was saved.
    """
    # Want to always have accuracy, so add it to the metrics if it is not
    # already present.
    if "Accuracy" not in metrics:
        metrics.append("Accuracy")

    logger.info(f"Evaluating predictions for {task}.")

    targets = []
    predictions = []
    input_lens = []
    choices_probs = {}
    pred_dicts = map(lambda l: json.loads(l), prediction_path.read_text('utf-8').splitlines(False))
    pbar = tqdm(desc="Reading")
    for line in pred_dicts:
        target = line['target']
        targets.append(target)
        predictions.append(line['prediction'][0])
        input_lens.append(len(line['input']))
        if line['choice_logits']:
            if target not in choices_probs:
                choices_probs[target] = {}
            for choice, prob in line['choice_logits'].items():
                if choice not in choices_probs[target]:
                    choices_probs[target][choice] = [prob]
                else:
                    choices_probs[target][choice].append(prob)
        pbar.update()
    pbar.close()

    logger.info("Final Metrics:")
    final_metrics = {}
    for m in metrics:
        for k, v in METRICS_DICT[m](targets, predictions).items():
            met_name = f"{k}:"
            logger.info(f"{met_name:>20} {v:.3f}")
            final_metrics[k] = v

    final_metrics['input_len/mean'] = np.mean(input_lens)
    final_metrics['input_len/median'] = np.median(input_lens)
    final_metrics['input_len/std'] = np.std(input_lens)

    target_lens = list(map(len, targets))
    final_metrics['target_len/mean'] = np.mean(target_lens)
    final_metrics['target_len/median'] = np.median(target_lens)
    final_metrics['target_len/std'] = np.std(target_lens)

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return metrics_file


def evaluate_dataset_with_prompt(
        task: str,
        dataset: Dataset,
        prompt: Template,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
        results_path: Path,
        batch_size: int,
        num_beams: int,
        force_generation: bool = False,
        length_normalization: bool = False,
        num_proc: int = 1
):
    tokenized, original, prompt = preprocess_dataset(
        task=task,
        dataset=dataset,
        tokenizer=tokenizer,
        prompt=prompt,
        num_proc=num_proc
    )

    choices = prompt.answer_choices
    logger.info(f"Choices found: {choices}")
    max_choices_found = max(map(len, original['choices']))
    min_choices_found = min(map(len, original['choices']))
    logger.info(f"Max # Choices found: {max_choices_found}")
    logger.info(f"Min # Choices found: {min_choices_found}")
    if max_choices_found != min_choices_found:
        logger.error("Variable number of choices found across examples. This is not supported.")
        raise ValueError("Variable number of choices found across examples. This is not supported.")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4,
        max_length=1024,
        padding='longest',
        label_pad_token_id=tokenizer.pad_token_id
    )

    data_loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False
    )

    logger.info(f"Max label length is {max(tokenized['labels_len'])}")
    logger.info(f"Max Input length is {max(tokenized['input_len'])}")
    device = torch.device(0)

    # TODO(gabeorlanski): Move generation into its own thing
    if choices is None or force_generation:
        result_file = generate_prediction_sequences(
            out_path=results_path,
            data_loader=data_loader,
            model=model,
            device=device,
            tokenizer=tokenizer,
            max_gen_len=max(tokenized['labels_len']) + 32,
            generator_kwargs={
                "num_beams"           : num_beams,
                "num_return_sequences": num_beams
            }
        )
    else:
        result_file = generate_predictions_choices(
            out_path=results_path,
            data_loader=data_loader,
            model=model,
            device=device,
            tokenizer=tokenizer,
            source_dataset=original,
            length_normalize=length_normalization
        )
    logger.info("Finished generating the dataset with the prompt.")
    logger.info(f"Beginning evaluation of the predictions.")
    evaluate(
        task,
        result_file,
        metrics=prompt.metadata.metrics or ["Accuracy"],
        out_path=results_path
    )
    return original
