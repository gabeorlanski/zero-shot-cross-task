#!/bin/bash
# Run the experiments

python experiment.py +run_name="CrossTask" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B"




