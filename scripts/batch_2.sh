#!/bin/bash
# Run the experiments
python experiment.py +run_name="CT_TNF" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['True','Neither','False']"


python experiment.py +run_name="CT_TNF" task=cb +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['True','Neither','False']"

python experiment.py +run_name="CT_ASN" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Always','Sometimes','Never']"



python experiment.py +run_name="CT_ASN" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Always','Sometimes','Never']"

python experiment.py +run_name="CT_INC" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Imply','Neither','Contradict']"



python experiment.py +run_name="CT_INC" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=28 \
  "task.preprocessor.choices=['Imply','Neither','Contradict']"
