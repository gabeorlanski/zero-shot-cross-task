import logging
from typing import Dict
from tqdm import tqdm
import torch
from t5.evaluation import metrics as mt
import json
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Model
import string

from src.preprocessing import preprocess_dataset

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
        choices
):
    """
    Generate predictions when using answer choices.

    Args:
        out_path (Path): Path to save to.
        data_loader (torch.utils.data.DataLoader): DataLoader to use.
        model: Model to use.
        tokenizer: Tokenizer to use.
        device: device to use.
        choices: Choices to use.

    Returns:
        Path where predictions were saved.
    """
    logger.info(f"Starting Generation")
    pred_path = out_path.joinpath('predictions.jsonl')
    pred_file = pred_path.open('w', encoding="utf-8")
    data_iterator = tqdm(data_loader, desc="Generating")
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError(f"Choices must be a list with at least two elements. got {choices}")

    # We tokenize the choices here as there is a chance that the choice itself
    # may be OOV or have more than one token, thus we handle those cases by
    # simply tokenizing the choices.
    choices_ids = tokenizer.convert_tokens_to_ids(choices)

    if any(map(
            lambda c: len(tokenizer(c, add_special_tokens=False)['input_ids']) > 1,
            choices)
    ):
        logger.error(f"Choices '{choices}' has a choice that is more than one "
                     f"token long. Not clear on how to handle that.")
        raise ValueError("No idea how to handle this case ATM.")

    for batch in data_iterator:
        logger.debug(f"Got batch with shape {batch['input_ids'].shape}")
        input_ids = batch['input_ids'].to(device)
        generated = model(
            input_ids=input_ids,
            attention_mask=batch['attention_mask'].to(device),
            labels=input_ids
        )

        # We only take the logits of the first token.
        choice_log_prob = generated.logits[:, 0, choices_ids]
        source = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        gold = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

        logger.debug("Saving JSON lines for batch")
        for i, target in enumerate(gold):
            ex_choice_prob = choice_log_prob[i, :]
            prediction_choice = choices[ex_choice_prob.argmax()]

            pred_file.write(serialize_prediction(
                [prediction_choice],
                target,
                source[i],
                {c: p.item() for c, p in zip(choices, ex_choice_prob)}
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


def evaluate_dataset_with_prompt(
        experiment_name,
        task,
        dataset,
        prompt_task,
        prompt_name,
        model_name,
        results_path,
        batch_size,
        use_base_model,
        num_beams,
        force_generation=False
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized, prompt = preprocess_dataset(
        task=experiment_name,
        dataset=dataset,
        tokenizer=tokenizer,
        prompt_task=prompt_task,
        prompt_name=prompt_name
    )

    # TODO(gabeorlanski): Make this work for non-fixed choice
    choices = prompt.get_fixed_answer_choices_list()

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
    device = torch.device(0)
    if use_base_model:
        model_cls = T5Model
    else:
        model_cls = T5ForConditionalGeneration
    model = model_cls.from_pretrained(model_name).to(device)
    model.eval()

    if choices is None or force_generation:
        result_file = generate_prediction_sequences(
            out_path=results_path,
            data_loader=data_loader,
            model=model,
            device=device,
            tokenizer=tokenizer,
            max_gen_len=max(tokenized['labels_len']) + 5,
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
            choices=choices
        )
    logger.info("Finished generating the dataset with the prompt.")
    logger.info(f"Beginning evaluation of the predictions.")
    evaluate(
        task,
        result_file,
        metrics=prompt.metadata.metrics or ["Accuracy"],
        out_path=results_path
    )
