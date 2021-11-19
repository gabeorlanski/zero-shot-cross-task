#!/bin/bash
# Run the experiments

python experiment.py +run_name="Baseline" task=cb +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=anli +split=dev_r1 model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=anli +split=dev_r2 model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=anli +split=dev_r3 model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=rte +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=wsc +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task="wsc.fixed" +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="Baseline" task=wic +split=validation model_name="bigscience/T0_3B"

python experiment.py +run_name="CrossTask" task=anli +split=dev_r1 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B"



python experiment.py +run_name="CrossTask" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="CrossTask" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="CrossTask" task=cb +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B"
