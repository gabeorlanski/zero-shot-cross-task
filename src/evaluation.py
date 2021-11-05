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


def generate(
        out_path,
        data_loader,
        model_name,
        tokenizer,
        max_gen_len,
        generator_kwargs: Dict = None
):
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch.device(0))
    logger.info(f"Starting Generation")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")

    data_iterator = tqdm(data_loader, desc="Generating")
    for batch in data_iterator:
        logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
        generated = model.generate(
            input_ids=batch['input_ids'].to(torch.device(0)),
            attention_mask=batch['attention_mask'].to(torch.device(0)),
            max_length=min(max_gen_len, 16),
            early_stopping=True,
            **(generator_kwargs or {})
        )

        source = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        gold = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        # Convert preds to a list of lists where each list only has a single
        # element so that we only need to handle the case where there are
        # multiple beams.
        generated = generated.reshape(
            (len(gold), generator_kwargs.get('num_return_sequences', 1), -1))

        # Pre initialize the preds nested array here because it saves time
        # later. Faster then appending.
        preds = [[None for _ in range(generated.shape[1])] for _ in range(len(gold))]
        for beam_num in range(generated.shape[1]):
            # Decode only this beam
            decoded_beams = tokenizer.batch_decode(
                generated[:, beam_num, :],
                skip_special_tokens=True
            )
            for i, seq in enumerate(decoded_beams):
                preds[i][beam_num] = seq

        logger.debug("Saving JSON lines for batch")
        for i, pred in enumerate(preds):
            pred_file.write(
                json.dumps({
                    "prediction": pred,
                    "target"    : gold[i],
                    "input"     : source[i]
                }) + '\n'
            )
    data_iterator.close()
    pred_file.close()
    return pred_path


def evaluate(task, prediction_path, metrics, out_path, expected_total):
    # Want to always have accuracy, so add it to the metrics if it is not
    # already present.
    if "Accuracy" not in metrics:
        metrics.append("Accuracy")
    logger.info(f"Evaluating predictions for {task}.")
    targets = []
    predictions = []
    m_trackers = defaultdict(list)
    pred_dicts = map(lambda l: json.loads(l), prediction_path.read_text('utf-8').splitlines(False))
    pbar = tqdm(total=expected_total, desc="Evaluating")
    for line in pred_dicts:
        targets.append(line['target'])
        predictions.append(line['prediction'][0])

        targets_mul = [line['target']] * len(line['prediction'])
        for m in metrics:
            # Rouge messes with a lot of stuff for some reason. So skip it.
            if m in ['ROUGE']:
                continue
            oracle = {}
            for x, y in zip(targets_mul, line['prediction']):
                for k, v in METRICS_DICT[m]([x], [y]).items():
                    oracle[k] = max(oracle.get(k, -1), v)
            for k, v in oracle.items():
                m_trackers[k].append(v)
        pbar.update()
    pbar.close()

    assert len(targets) == expected_total
    logger.info("Final Metrics:")
    final_metrics = {}
    for m in metrics:
        for k, v in METRICS_DICT[m](targets, predictions).items():
            met_name = f"{k}:"
            logger.info(f"{met_name:>20} {v:.3f}")
            final_metrics[k] = v

            if k not in m_trackers:
                continue
            met_name = f"oracle_{k}"

            final_metrics[met_name] = np.mean(m_trackers[k])
            logger.info(f"{met_name:>20} {final_metrics[met_name]:.3f}")

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return metrics_file
