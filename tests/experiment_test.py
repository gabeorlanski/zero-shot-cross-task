import pytest
from pathlib import Path
from src import experiment
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from datasets import set_caching_enabled
from unittest.mock import patch
from functools import partial
from promptsource.templates import DatasetTemplates


@pytest.mark.parametrize("force", [True, False], ids=["Force", "No Force"])
@pytest.mark.parametrize("existing_dir", ["A", "Prompt"], ids=["No Conflict", "Conflict"])
@pytest.mark.parametrize("subset", [None, "cb"], ids=["No Subset", "Subset"])
def test_single_experiment(tmpdir, force, existing_dir, subset):
    tmpdir_path = Path(tmpdir)

    cfg = OmegaConf.create({
        "task"       : "Test",
        "model_name" : "Test",
        "force"      : force,
        "batch_size" : 8,
        "beams"      : 4,
        "evaluation" : {
            "force_generation"    : False,
            "length_normalization": False,
        },
        "num_proc"   : 1,
        "cuda_device": 0

    })

    dataset_name = "anli" if not subset else "super_glue"
    split = "train[:10]" if subset else "train_r1[:10]"

    prompt = DatasetTemplates('anli')["does it follow that"]
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    set_caching_enabled(False)

    tmpdir_path.joinpath(existing_dir).mkdir(parents=True)

    if not force and existing_dir == "Prompt":
        with pytest.raises(ValueError):
            experiment.single_experiment(
                cfg,
                prompt=prompt,
                dataset_name=dataset_name,
                split=split,
                model=None,
                tokenizer=tokenizer,
                experiment_name="Test",
                prompt_file_name="Prompt",
                seeds={},
                subset=subset,
                working_dir=tmpdir_path
            )
        return

    with patch('src.experiment.evaluate') as eval_func:
        with patch('src.experiment.generate_prediction_sequences') as gen_func:
            with patch('src.experiment.generate_predictions_choices') as choice_func:
                gen_func.return_value = tmpdir_path
                choice_func.return_value = tmpdir_path
                ds, results_path = experiment.single_experiment(
                    cfg,
                    prompt=prompt,
                    dataset_name=dataset_name,
                    split=split,
                    model=None,
                    tokenizer=tokenizer,
                    experiment_name="Test",
                    prompt_file_name="Prompt",
                    seeds={},
                    subset=subset,
                    working_dir=tmpdir_path
                )
                assert eval_func.call_count == 1
                assert results_path.exists()
    set_caching_enabled(True)
