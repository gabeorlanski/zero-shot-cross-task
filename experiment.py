import logging
import hydra
from omegaconf import DictConfig
from unidecode import unidecode
from pathlib import Path

from src.experiment import run_experiments
from src.prompt_map import load_prompts, load_general_prompts

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

    if not cfg.get('use_general_prompts'):
        prompts_to_use = load_prompts(
            prompt_task, categories, prompt_filter_kwargs=cfg['prompt_filter']
        )
    else:
        prompts_to_use = load_general_prompts(
            prompt_dir=PROJECT_ROOT.joinpath(cfg["general_prompts"]['dir']),
            prompt_cfg=task_cfg['general_prompts'],
            category_filter=cfg['general_prompts']['category_filter'],
            prompt_filter_kwargs=cfg['prompt_filter'],
            answer_filter=cfg['general_prompts']['answer_filter']
        )
    if cfg["debug"]:
        logger.warning(f"Debugging enbaled, only using a single prompt")
        prompts_to_use = prompts_to_use[:1]
    logger.info(f"Prompts to use are: {', '.join(p['name'] for _, _, p in prompts_to_use)}")
    logger.info(f"{len(prompts_to_use)} total prompts ")

    run_experiments(
        cfg=cfg,
        prompts_to_use=prompts_to_use,
        dataset_name=dataset_name,
        split_name=split,
        task_name=task_name,
        verbose_name=verbose_name,
        categories=categories,
        seeds=seeds
    )


if __name__ == '__main__':
    run()
