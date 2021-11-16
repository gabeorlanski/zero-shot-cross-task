#!/bin/bash
# Run the experiments
python experiment.py +run_name="AC" task=cb +split=validation \
  num_proc=4 \
  debug=False \
  cuda_device=1 \
  use_general_prompts=True \
  prompt_filter.name_list=["does it follow that","can we infer","justified in saying","should assume MCQ","support"] \
  evaluation.length_normalization=False \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r1 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  cuda_device=1 \
  prompt_filter.name_list=["does it follow that","can we infer","justified in saying","should assume MCQ","support"] \
  evaluation.length_normalization=False \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_filter.name_list=["does it follow that","can we infer","justified in saying","should assume MCQ","support"] \
  debug=False \
  cuda_device=1 \
  use_general_prompts=True \
  evaluation.length_normalization=False \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_filter.name_list=["does it follow that","can we infer","justified in saying","should assume MCQ","support"] \
  debug=False \
  use_general_prompts=True \
  cuda_device=1 \
  evaluation.length_normalization=False \
  model_name="bigscience/T0_3B"