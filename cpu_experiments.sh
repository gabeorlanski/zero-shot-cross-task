#!/bin/bash
# Run the experiments
python experiment.py +run_name="AC" task=cb +split=validation \
  num_proc=4 \
  debug=False \
  cuda_device=-1 \
  use_general_prompts=False \
  prompt_filter.name_list=["does it follow that"] \
  evaluation.length_normalization=False \
  model_name="bigscience/T0" \
  disable_tracking=True