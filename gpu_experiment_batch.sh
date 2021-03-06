#!/bin/bash
# Run the experiments

python experiment.py +run_name="CTBase" task=anli +split=dev_r1 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True

python experiment.py +run_name="CTBase" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True

python experiment.py +run_name="CTBase" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True

python experiment.py +run_name="CTBase" task=anli +split=dev_r3 num_proc=4 prompt_experiment_mode=cross_task prompt_path=prompts/general_fixed_choice.yaml model_name="bigscience/T0_3B" evaluation.length_normalization=True batch_size=32 prompt_count=2 disable_tracking=True

