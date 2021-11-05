import logging
from collections import Counter

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


def evaluate(task, out_path, data_loader, model_name, tokenizer, metrics):

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch.device(0))
    logger.info(f"Starting Generation")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")
    batches_seen = 0

    # Create a metric tracker for keeping track of metrics we use.
    metric_tracker = Counter()

    data_iterator = tqdm(data_loader)
    for batch in data_iterator:
        logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
        generated = model.generate(
            input_ids=batch['input_ids'].to(torch.device(0)),
            attention_mask=batch['attention_mask'].to(torch.device(0)),
            max_length=128,
            early_stopping=True,
        )

        source = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        gold = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        logger.debug(f"Calculating metrics {metrics}")
        for m in metrics:
            batch_metrics = METRICS_DICT[m](gold, preds)
            for k, v in batch_metrics.items():
                metric_tracker[k] += v

        logger.debug("Saving JSON lines")
        for i, pred in enumerate(preds):
            if len(gold[i]) > 128:
                raise ValueError("Longer Outputs then expected.")
            pred_file.write(
                json.dumps({
                    "prediction": pred,
                    "target"    : gold[i],
                    "input"     : source[i]
                }) + '\n'
            )
        batches_seen += 1

        metric_str = ""
        for name, value in metric_tracker.items():
            metric_str += f"{name}: {value / batches_seen:.3f} "
        data_iterator.set_description(
            metric_str,
            refresh=True
        )
    data_iterator.close()
    logger.info(f"Final scores for {task}:")
    final_metrics = {}
    for k, v in metric_tracker.items():
        final_metrics[k] = v / batches_seen
        logger.info(f"\t{k}: {v / batches_seen:.3f}")
    pred_file.close()
    logger.info("Finished applying the prompt.")

    metrics_file = out_path.joinpath("metrics.json")
    logger.info(f"Saving metrics to '{metrics_file}'")
    with metrics_file.open('w', encoding='utf-8') as fp:
        fp.write(json.dumps(final_metrics, indent=True))

    return pred_path
