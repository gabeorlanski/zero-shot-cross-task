import logging
from collections import Counter, defaultdict
from typing import Dict
from math import isclose
import numpy as np
from tqdm import tqdm
import torch
from t5.data.glue_utils import get_glue_metric, get_super_glue_metric
from t5.evaluation import metrics as mt
import json
from transformers import T5ForConditionalGeneration

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


def generate_prediction_sequences(
        out_path,
        data_loader,
        model,
        tokenizer,
        device,
        max_gen_len,
        generator_kwargs: Dict = None
):
    logger.info(f"Starting Generation")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")
    generator_kwargs = generator_kwargs or {}
    data_iterator = tqdm(data_loader, desc="Generating")
    for batch in data_iterator:
        logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
        generated = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_length=min(max_gen_len, 16),
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
            pred_file.write(
                json.dumps({
                    "prediction": preds[i * num_beams:(i + 1) * num_beams],
                    "target"    : target,
                    "input"     : source[i],
                    "choices"   : []
                }) + '\n'
            )
    data_iterator.close()
    pred_file.close()
    return pred_path


def evaluate(task, prediction_path, metrics, out_path):
    # Want to always have accuracy, so add it to the metrics if it is not
    # already present.
    if "Accuracy" not in metrics:
        metrics.append("Accuracy")

    logger.info(f"Evaluating predictions for {task}.")

    targets = []
    predictions = []
    pred_dicts = map(lambda l: json.loads(l), prediction_path.read_text('utf-8').splitlines(False))
    pbar = tqdm(desc="Reading")
    for line in pred_dicts:
        targets.append(line['target'])
        predictions.append(line['prediction'][0])
        pbar.update()
    pbar.close()

    logger.info("Final Metrics:")
    final_metrics = {}
    for m in metrics:
        for k, v in METRICS_DICT[m](targets, predictions).items():
            met_name = f"{k}:"
            logger.info(f"{met_name:>20} {v:.3f}")
            final_metrics[k] = v

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return metrics_file
