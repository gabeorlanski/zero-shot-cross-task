#!/bin/bash
# Run the experiments
rm -rf outputs/cb.validation
rm -rf outputs/anli.dev_r1
rm -rf outputs/anli.dev_r2
rm -rf outputs/anli.dev_r3

python experiment.py task=cb +split=validation \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py task=anli +split=dev_r1 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py task=anli +split=dev_r2 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py task=anli +split=dev_r3 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"
