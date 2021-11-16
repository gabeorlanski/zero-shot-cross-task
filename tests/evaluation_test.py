import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from unittest.mock import MagicMock
import torch
import math
import json
from src import evaluation
from src.preprocessing import preprocess_dataset
from promptsource.templates import DatasetTemplates, Template


def test_generate_prediction_sequences(tmpdir):
    ds = load_dataset("anli", split="train_r1[:16]")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def tok(p, h, ex_idx):
        labels = tokenizer(h)
        source = tokenizer(p)
        return {
            "idx"      : ex_idx, "labels": labels['input_ids'],
            "input_len": sum(source['attention_mask']), **source
        }

    ds = ds.map(  # type: ignore
        tok,
        input_columns=['premise', 'hypothesis'],
        remove_columns=ds.column_names,
        with_indices=True
    ).sort("input_len")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4,
        max_length=1024,
        padding='longest',
        label_pad_token_id=tokenizer.pad_token_id
    )

    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        collate_fn=collator,
        shuffle=False
    )

    model = T5ForConditionalGeneration.from_pretrained('patrickvonplaten/t5-tiny-random')
    model.eval()
    model.generate = MagicMock()

    expected_tokens = tokenizer(
        ["This is a test for evaluation", "This is a second beam"],
        padding="longest"
    )['input_ids']
    model.generate.return_value = torch.Tensor(expected_tokens).long().reshape(2, -1)

    res_path = evaluation.generate_prediction_sequences(
        Path(tmpdir),
        data_loader,
        model,
        tokenizer,
        torch.device("cpu"),
        5,
        {'num_return_sequences': 2}
    )

    assert res_path.stem == "predictions"
    assert res_path.exists()

    result = list(map(json.loads, res_path.read_text("utf-8").splitlines(False)))
    assert len(result) == len(ds)

    for i, v in enumerate(ds):
        assert result[i]['prediction'] == ["This is a test for evaluation",
                                           "This is a second beam"]

        assert result[i]['target'] == tokenizer.decode(v['labels'], skip_special_tokens=True)
        assert result[i]['input'] == tokenizer.decode(v['input_ids'], skip_special_tokens=True)


@pytest.mark.parametrize("prompt_name", ["claim true/false/inconclusive", "guaranteed true"])
@pytest.mark.parametrize("length_normalized", [True, False],
                         ids=["W/Normalization", "No Normalization"])
@pytest.mark.parametrize("force_not_fixed", [True, False],
                         ids=["Force Not Fixed", "Not Fixed"])
def test_generate_prediction_choices(tmpdir, prompt_name, length_normalized, force_not_fixed):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    original_dataset = load_dataset("anli", split="train_r1[:16]")
    prompt:Template = DatasetTemplates('anli')[prompt_name]
    choices = prompt.get_fixed_answer_choices_list()

    tokenized, ds, prompt = preprocess_dataset(
        "Testing",
        original_dataset,
        tokenizer,
        prompt,
        num_proc=1
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=4,
        max_length=1024,
        padding='longest',
        label_pad_token_id=tokenizer.pad_token_id
    )

    data_loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=16,
        collate_fn=collator,
        shuffle=False
    )

    choice_ids = list(map(
        lambda c: tokenizer(c, add_special_tokens=False)['input_ids'],
        choices)
    )
    single_batch = collator(list(tokenized))
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model.eval()

    expected_logits = model(
        input_ids=single_batch['input_ids'],
        attention_mask=single_batch['attention_mask'],
        labels=single_batch['choices_tokenized']
    ).logits

    expected_choice_probs = torch.zeros((expected_logits.shape[0], len(choices)))
    for choice, tokens in enumerate(choice_ids):
        for t in tokens:
            expected_choice_probs[:, choice] += expected_logits[:, :, t].sum(-1)
        if length_normalized:
            expected_choice_probs[:, choice] /= len(tokens)

    res_path = evaluation.generate_predictions_choices(
        Path(tmpdir),
        data_loader,
        model,
        tokenizer,
        torch.device("cpu"),
        source_dataset=ds,
        length_normalize=length_normalized,
        force_not_fixed_choice=force_not_fixed
    )

    assert res_path.stem == "predictions"
    assert res_path.exists()

    result = list(map(json.loads, res_path.read_text("utf-8").splitlines(False)))
    assert len(result) == len(ds)

    for i, v in enumerate(tokenized):
        assert set(result[i]['choice_logits'].keys()) == set(choices)
        assert result[i]['prediction'] == [
            max(choices, key=lambda x: result[i]['choice_logits'][x])
        ]

        assert result[i]['target'] == tokenizer.decode(v['labels'], skip_special_tokens=True)
        assert result[i]['input'] == tokenizer.decode(v['input_ids'], skip_special_tokens=True)

        for j, choice in enumerate(choices):
            expected = expected_choice_probs[i, j].item()
            assert math.isclose(
                result[i]['choice_logits'][choice],
                expected,
                rel_tol=1e-4
            ), f"{result[i]['choice_logits'][choice]:.2f} != {expected:.2f}"


def test_evaluate(tmpdir):
    pred_file = Path(tmpdir).joinpath('predictions.jsonl')
    preds = "A G D E E A Q".split()
    targets = "A B C D E F G".split()
    with pred_file.open('w', encoding='utf-8') as f:
        for p, t in zip(preds, targets):
            f.write(json.dumps(
                {"prediction": [p, "NONE"], "target": t, "input": t, "choice_logits": {}}) + '\n')

    res_file = evaluation.evaluate("test", pred_file, [], Path(tmpdir))
    assert res_file.stem == "metrics"
    assert res_file.exists

    result = json.loads(res_file.read_text('utf-8'))
    assert result == {
        'input_len/mean'   : 1.0,
        'input_len/median' : 1.0,
        'input_len/std'    : 0.0,
        'target_len/mean'  : 1.0,
        'target_len/median': 1.0,
        'target_len/std'   : 0.0,
        **evaluation.METRICS_DICT["Accuracy"](targets, preds)
    }
