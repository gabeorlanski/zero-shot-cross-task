import pandas as pd
import wandb
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()
if 'scripts' in str(PROJECT_ROOT):
    while 'scripts' in str(PROJECT_ROOT):
        PROJECT_ROOT = PROJECT_ROOT.parent


def download_wandb_runs():
    out_path = PROJECT_ROOT.joinpath('data')
    run_csv_path = out_path.joinpath('runs.csv')
    summary_keys_path = out_path.joinpath('summary_keys.txt')
    print(f'Downloading to {out_path}')
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("gabeorlanski/zero-shot-eval")
    print(f"{len(runs)} runs found")
    records = []
    summary_keys = set()
    for run in tqdm(runs, desc='Aligning data'):
        record_dict = {}
        run_summary = {k: v for k, v in run.summary._json_dict.items()
                       if not k.startswith('_')}
        summary_keys.update(set(run_summary))
        record_dict.update(run_summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        record_dict.update({k: v for k, v in run.config.items()
                            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        record_dict['name'] = run.name
        record_dict['group'] = run.group
        records.append(record_dict)

    runs_df = pd.DataFrame.from_records(records)
    with summary_keys_path.open('w', encoding='utf-8') as f:
        for k in summary_keys:
            f.write(k + '\n')

    runs_df.to_csv(run_csv_path)


if __name__ == '__main__':
    download_wandb_runs()
