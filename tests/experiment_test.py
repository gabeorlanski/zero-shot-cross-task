import pytest
from pathlib import Path
from src import experiment
from omegaconf import OmegaConf
from unittest.mock import patch
from functools import partial


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

    experiment_partial = partial(
        experiment.single_experiment,
        prompt=None,
        dataset_name=dataset_name,
        split=split,
        model=None,
        tokenizer=None,
        experiment_name="Test",
        prompt_file_name="Prompt",
        seeds={},
        subset=subset,
        working_dir=tmpdir_path
    )

    tmpdir_path.joinpath(existing_dir).mkdir(parents=True)

    if not force and existing_dir == "Prompt":
        with pytest.raises(ValueError):
            experiment_partial(cfg)
        return

    with patch('src.experiment.evaluate_dataset_with_prompt') as eval_func:
        eval_func.return_value = "X"
        ds, results_path = experiment_partial(cfg)
        assert eval_func.call_count == 1
        call_args = list(map(list, eval_func.call_args_list))[0][1]
        assert call_args["task"] == cfg['task']
        assert call_args["results_path"] == results_path
        assert results_path.exists()
    assert ds == "X"
