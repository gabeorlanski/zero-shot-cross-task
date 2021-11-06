import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from unittest.mock import MagicMock
import torch

import json
from src import evaluation


def test_generate_prediction_sequences(tmpdir):
    ds = load_dataset("anli", split="train_r1[:16]")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    def tok(p, h):
        labels = tokenizer(h)
        source = tokenizer(p)
        return {"labels": labels['input_ids'], "input_len": sum(source['attention_mask']), **source}

    ds = ds.map(  # type: ignore
        tok,
        input_columns=['premise', 'hypothesis'],
        remove_columns=ds.column_names
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


def test_evaluate(tmpdir):
    pred_file = Path(tmpdir).joinpath('predictions.jsonl')
    preds = "A G D E E A Q".split()
    targets = "A B C D E F G".split()
    with pred_file.open('w', encoding='utf-8') as f:
        for p, t in zip(preds, targets):
            f.write(json.dumps({"prediction": [p, "NONE"], "target": t}) + '\n')

    res_file = evaluation.evaluate("test", pred_file, [], Path(tmpdir))
    assert res_file.stem == "metrics"
    assert res_file.exists

    result = json.loads(res_file.read_text('utf-8'))
    assert result == evaluation.METRICS_DICT["Accuracy"](targets, preds)
