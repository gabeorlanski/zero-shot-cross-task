#!/bin/bash
# Run the experiments
python experiment.py +run_name="T5" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="t5-3b" \
  evaluation.length_normalization=False \
  batch_size=6 \
  dont_add_extra_text=True \
  prompt_tasks=['craigslist_bargains']