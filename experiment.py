import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from unidecode import unidecode
from pathlib import Path

from datasets import set_caching_enabled
from src.experiment import run_experiments
from src.prompt_map import load_prompts, load_answer_choice_experiment_prompts, \
    load_generalized_prompts
from src.preprocessors import FixedChoiceTaskPreprocessor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()


@hydra.main(config_path='configs', config_name="config")
def run(cfg: DictConfig):
    """
    Run an experiment

    Args:
        cfg: Run configs
    """
    logger.info(f"Starting Experiment.")
    # Setup the seeds and logging
    seeds = dict(
        seed=1966,
        numpy_seed=1991,
        torch_seed=1999
    )

    if cfg['evaluation']['length_normalization'] and cfg['evaluation']['force_generation']:
        raise ValueError("Cannot have both 'length_normalization' and 'force_generation'")

    # Get the task config
    task_cfg = cfg['task']
    split = cfg['split']

    # Get names
    task_name = unidecode(task_cfg['name'])
    verbose_name = task_cfg.get('verbose_name', task_name)
    dataset_name = task_cfg.get("parent_dataset", task_name)
    categories = task_cfg['category']

    set_caching_enabled(not cfg['disable_caching'])

    if isinstance(categories, str):
        categories = [categories]
    else:
        # To remove the annoying ListConfig object.
        categories = list(categories)

    # Some task names will be different then their actual name from HuggingFace
    # due to being a subset of a task (e.g., super_glue/rte)
    if task_name != dataset_name:
        prompt_task = unidecode(f"{dataset_name}/{task_name}")
    else:
        prompt_task = task_name

    logger.info(f"Starting experiment with task {verbose_name}")
    logger.info(f"Using Split {split}")

    if cfg['cuda_device'] < 0:
        cfg['cuda_device'] = 'cpu'
    prompt_experiment_mode = cfg['prompt_experiment_mode']
    preprocessor = None  # type: ignore

    if prompt_experiment_mode == "answer_choices":
        prompts_to_use = load_answer_choice_experiment_prompts(
            prompt_dir=PROJECT_ROOT.joinpath(cfg["general_prompts"]['dir']),
            prompt_cfg=task_cfg['general_prompts'],
            category_filter=cfg['general_prompts']['category_filter'],
            prompt_filter_kwargs=cfg['prompt_filter'],
            answer_filter=cfg['general_prompts']['answer_filter']
        )
    elif prompt_experiment_mode == "cross_task":
        if cfg['prompt_path'] is None:
            raise ValueError("Need a path to prompts for cross task")
        preprocessor_args = OmegaConf.to_object(task_cfg['preprocessor'])
        preprocessor_cls = FixedChoiceTaskPreprocessor.by_name(preprocessor_args.pop('name'))
        preprocessor: FixedChoiceTaskPreprocessor = preprocessor_cls(**preprocessor_args)

        prompts_to_use = load_generalized_prompts(
            PROJECT_ROOT.joinpath(cfg['prompt_path']),
            task_name=task_name,
            choices=preprocessor.choices,
            choice_str=preprocessor.choice_string,
            mcq_choice_str=preprocessor.mcq_choice_string,
            prompt_filter_kwargs=cfg['prompt_filter'],
        )

    else:
        prompts_to_use = load_prompts(
            prompt_task, categories, prompt_filter_kwargs=cfg['prompt_filter']
        )

    if cfg["prompt_count"] is not None and cfg["prompt_count"] > 0:
        logger.warning(f"Debugging enbaled, only using a single prompt")
        prompts_to_use = prompts_to_use[:cfg['prompt_count']]
    logger.info(f"Prompts to use are: {', '.join(p['name'] for _, _, p in prompts_to_use)}")
    logger.info(f"{len(prompts_to_use)} total prompts ")

    run_experiments(
        cfg=cfg,
        prompts_to_use=prompts_to_use,
        dataset_name=dataset_name,
        preprocessor=preprocessor,
        split_name=split,
        task_name=task_name,
        verbose_name=verbose_name,
        categories=categories,
        seeds=seeds
    )


if __name__ == '__main__':
    run()
