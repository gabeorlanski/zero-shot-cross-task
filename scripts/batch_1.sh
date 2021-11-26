#!/bin/bash
# Run the experiments
python experiment.py +run_name="CT_INC" task=anli +split=dev_r1 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Imply','Neither','Contradicts']"

python experiment.py +run_name="CT_INC" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Imply','Neither','Contradicts']"

