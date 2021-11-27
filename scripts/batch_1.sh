#!/bin/bash
# Run the experiments

python experiment.py +run_name="CTBase" task=cb +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"

python experiment.py +run_name="CTBase" task=anli +split=dev_r1 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"

python experiment.py +run_name="CTBase" task=wic +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "prompt_tasks=['aqua_rat/raw','math_qa','craigslist_bargains']"


python experiment.py +run_name="CTBase" task=aqua +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=12

python experiment.py +run_name="T0" task=aqua +split=validation \
  num_proc=4 \
  prompt_experiment_mode=default \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=False \
  batch_size=12


python experiment.py +run_name="T5" task=aqua +split=validation \
  num_proc=4 \
  prompt_experiment_mode=default \
  model_name="t5-3b" \
  evaluation.length_normalization=False \
  batch_size=6



