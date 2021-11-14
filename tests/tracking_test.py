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


def test_create_predictions_df(tmpdir):
    pred_path = Path(tmpdir).joinpath("preds.jsonl")
    with pred_path.open('w', encoding='utf-8') as f:
        data = [
            {
                "id"           : 23, "prediction": [",,,,,,,, that's very nice.,,,,,"],
                "target"       : "No",
                "input"        : "Abc",
                "choice_logits": {}
            },
            {
                "id"           : 50, "prediction": [
                ",,, I don't think the judge should just make the decision alone, but I don't think the judge should just make the decision",
                "C"],
                "target"       : "No",
                "input"        : "DEF",
                "choice_logits": {}
            },
            {
                "id"           : 35,
                "prediction"   : [
                    ", uh, and it's interesting because, uh, so many problems, I work in a high school, are that kids don"
                ],
                "target"       : "Yes",
                "input"        : "GHI",
                "choice_logits": {}
            }
        ]
        for l in data:
            f.write(json.dumps(l) + "\n")

    result = tracking.create_predictions_df(pred_path)

    expected = pd.DataFrame.from_records(
        [{
            "id"           : 23,
            "prediction"   : ",,,,,,,, that's very nice.,,,,,",
            "other_beams"  : [],
            "target"       : "No",
            "input"        : "Abc",
            "choice_logits": {}
        }, {
            "id"           : 50,
            "prediction"   :
                ",,, I don't think the judge should just make the decision alone, but I don't think the judge should just make the decision",
            "other_beams"  : ["C"],
            "target"       : "No",
            "input"        : "DEF",
            "choice_logits": {}
        }, {
            "id"           : 35,
            "prediction"   :
                ", uh, and it's interesting because, uh, so many problems, I work in a high school, are that kids don"
            ,
            "other_beams"  : [],
            "target"       : "Yes",
            "input"        : "GHI",
            "choice_logits": {}
        }
        ]
    )
    result = result.reindex(sorted(result.columns), axis=1)
    expected = expected.reindex(sorted(expected.columns), axis=1).sort_values(by=['id'])
    pd._testing.assert_frame_equal(result, expected)
