#!/bin/bash
# Run the experiments
python experiment.py +run_name="CTBase" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"

python experiment.py +run_name="CTBase" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"


python experiment.py +run_name="CTBase" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=12 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"


python experiment.py +run_name="CTBase" task=rte +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"


python experiment.py +run_name="T0" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=False \
  batch_size=12 \
  "prompt_tasks=['craigslist_bargains']"


python experiment.py +run_name="T5" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="t5-3b" \
  evaluation.length_normalization=False \
  batch_size=6 \
  "prompt_tasks=['craigslist_bargains']"