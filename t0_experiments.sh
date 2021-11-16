#!/bin/bash
# Run the experiments
python experiment.py +run_name="AC" task=cb +split=validation \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  evaluation.length_normalization=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r1 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  evaluation.length_normalization=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r2 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  evaluation.length_normalization=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r3 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  evaluation.length_normalization=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=cb +split=validation num_proc=4 \
  prompt_filter.name_list=["does it follow that","can we infer","justified in saying","should assume MCQ","support"] \
  use_general_prompts=True \
  evaluation.force_generation=True \
  model_name="bigscience/T0_3B"



