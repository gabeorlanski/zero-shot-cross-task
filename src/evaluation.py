import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
from t5.evaluation import metrics as mt
import json
import numpy as np
from src.common import all_equal
import seqio

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
        idx,
        prediction,
        target,
        input_seq,
        choice_logits=None
) -> str:
    """
    Helper function to turn predictions into writeable string.

    Args:
        idx (int): id of the prediction
        prediction (list): List of prediction strings.
        target (str): The target string.
        input_seq (str): The input string.
        choice_logits (dict): The choices (if exist) dict with log probabilities.

    """
    return json.dumps({
        "id"           : idx,
        "prediction"   : prediction,
        "target"       : target,
        "input"        : input_seq,
        "choice_logits": choice_logits or {}
    })


def score_choice(logits, choice, length_normalize: bool = False):
    return torch.sum(logits[:, :, choice], (1, 2)) / (1 if not length_normalize else len(choice))


def generate_predictions_choices(
        data_loader,
        model,
        device,
        length_normalize=False,
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
    data_iterator = tqdm(data_loader, desc="Generating")

    predictions = defaultdict(list)
    with torch.no_grad():
        for batch in data_iterator:
            if batch['idx'].shape[0] != 1:
                raise ValueError('Only batch size 1 supported at the moment.')

            generated = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['labels'].to(device)
            )
            predictions['targets'].append(
                (batch['idx'][0].tolist(), batch['is_correct'][0].tolist(), 1.0)
            )
            predictions['scores'].extend(
                score_choice(
                    generated.logits,
                    batch['target'][0].tolist(),
                    length_normalize=length_normalize
                ).tolist()
            )

    data_iterator.close()
    # pred_file.close()
    return predictions


def evaluate(task, predictions, source_dataset, metrics, out_path):
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

    # This program is only made for fixed choice tasks. So we can assume all #
    # of choices will stay the same.
    num_choices = len(source_dataset[0]['choices'])

    final_metrics = mt.rank_classification(
        predictions['targets'],
        predictions['scores'],
        num_classes=num_choices
    )

    aligned_preds = [[None] * num_choices for _ in range(len(source_dataset))]
    aligned_targets = [None] * len(source_dataset)

    # Align the predictions with the targets and the source data so that they
    # can be saved to a file. Also save them to their own arrays so that we can
    # calculate per class F1
    for target, score in zip(predictions['targets'], predictions['scores']):
        (idx, choice_idx), is_correct, _ = target
        if is_correct:
            aligned_targets[idx] = choice_idx
        aligned_preds[idx][choice_idx] = score

    f1_metrics = mt.sklearn_metrics_wrapper(
        "fbeta_score",
        metric_dict_str="f1_by_class",
        metric_post_process_fn=lambda x: 100 * x,
        beta=1,
        labels=range(num_choices),
        average=None
    )(
        np.array(aligned_preds).argmax(-1), np.array(aligned_targets)
    )

    for l, f1 in enumerate(f1_metrics['f1_by_class']):
        final_metrics[f"f1_choice_{l}"] = f1
    input_lens = list(map(len, source_dataset['prompt']))
    final_metrics['input_len/mean'] = np.mean(input_lens)  # type: ignore
    final_metrics['input_len/median'] = np.median(input_lens)  # type: ignore
    final_metrics['input_len/std'] = np.std(input_lens)  # type: ignore

    for k, v in final_metrics.items():
        logger.info(f"{k:>20}: {v:.2f}")

    pred_file = out_path.joinpath('predictions.jsonl')
    with pred_file.open('w', encoding='utf-8') as pred_fp:
        for i, preds in enumerate(aligned_preds):
            ex = source_dataset[i]
            raise NotImplementedError()

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return metrics_file, pred_file
