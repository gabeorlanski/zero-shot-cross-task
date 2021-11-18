import json

import pandas as pd
import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset
from pathlib import Path
from omegaconf import OmegaConf
from src.common.util import PROJECT_ROOT
import shutil
from promptsource.templates import DatasetTemplates
from functools import partial
import yaml
from src import tracking
from src.common import sanitize_name


def test_get_metrics_for_wandb(tmpdir):
    pred_path = Path(tmpdir).joinpath("preds.jsonl")
    met_path = Path(tmpdir).joinpath("metrics.json")
    with met_path.open("w", encoding='utf-8') as f:
        json.dump({"accuracy": 100}, f)

    with pred_path.open('w', encoding='utf-8') as f:
        data = [
            {
                "id"           : 23, "prediction": ["A"],
                "target"       : "B",
                "input"        : "Abc",
                "choices"      : ["A", "B", "C"],
                "choice_logits": {"A": -1, "B": -4, "C": -5}
            },
            {
                "id"           : 50, "prediction": [
                ",,, I don't think the judge should just make the decision alone, but I don't think the judge should just make the decision",
                "C"],
                "target"       : "No",
                "input"        : "DEF",
                "choices"      : [],
                "choice_logits": {}
            },
            {
                "id"           : 35,
                "prediction"   : [
                    ", uh, and it's interesting because, uh, so many problems, I work in a high school, are that kids don"
                ],
                "target"       : "Yes",
                "input"        : "GHI",
                "choices"      : [],
                "choice_logits": {}
            }
        ]
        for l in data:
            f.write(json.dumps(l) + "\n")

    metrics, result = tracking.get_metrics_for_wandb(met_path, pred_path)

    assert set(metrics) == {
        "logits/c0_logit_mean", "logits/c1_logit_mean", "logits/c2_logit_mean",
        "logits/c0_logit_std", "logits/c1_logit_std", "logits/c2_logit_std",
        "logits/c0_logit_median", "logits/c1_logit_median", "logits/c2_logit_median",
        "accuracy"
    }
    assert metrics['accuracy'] == 100

    expected = pd.DataFrame.from_records(
        [{
            "id"          : 23,
            "prediction"  : "A",
            "other_beams" : [],
            "target"      : "B",
            "input"       : "Abc",
            "choice_count": 3,
            "choice_0"    : "A",
            "choice_1"    : "B",
            "choice_2"    : "C",
            "c0_logit"    : -1,
            "c1_logit"    : -4,
            "c2_logit"    : -5,
            "target_id"   : 1,
            "pred_id"     : 0,
            "correct"     : False

        }, {
            "id"          : 50,
            "prediction"  :
                ",,, I don't think the judge should just make the decision alone, but I don't think the judge should just make the decision",
            "other_beams" : ["C"],
            "target"      : "No",
            "input"       : "DEF",
            "choice_count": 0,
            "correct"     : False
        }, {
            "id"          : 35,
            "prediction"  :
                ", uh, and it's interesting because, uh, so many problems, I work in a high school, are that kids don"
            ,
            "other_beams" : [],
            "target"      : "Yes",
            "input"       : "GHI",
            "choice_count": 0,
            "correct"     : False
        }
        ]
    )
    result = result.reindex(sorted(result.columns), axis=1)
    expected = expected.reindex(sorted(expected.columns), axis=1).sort_values(by=['id'])
    pd._testing.assert_frame_equal(result, expected)
