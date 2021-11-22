#!/bin/bash
# Run the experiments
python experiment.py +run_name="T0" task=cb +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=anli +split=dev_r1 model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=anli +split=dev_r2 model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=anli +split=dev_r3 model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=rte +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=wsc +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task="wsc.fixed" +split=validation model_name="bigscience/T0_3B"
python experiment.py +run_name="T0" task=wic +split=validation model_name="bigscience/T0_3B"

python experiment.py +run_name="T5" task=cb +split=validation model_name="t5-3b"
python experiment.py +run_name="T5" task=anli +split=dev_r1 model_name="t5-3b"
python experiment.py +run_name="T5" task=anli +split=dev_r2 model_name="t5-3b"
python experiment.py +run_name="T5" task=anli +split=dev_r3 model_name="t5-3b"
python experiment.py +run_name="T5" task=rte +split=validation model_name="t5-3b"
python experiment.py +run_name="T5" task=wsc +split=validation model_name="t5-3b"
python experiment.py +run_name="T5" task="wsc.fixed" +split=validation model_name="t5-3b"
python experiment.py +run_name="T5" task=wic +split=validation model_name="t5-3b"


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

python experiment.py +run_name="CTBase" task=cb +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True

python experiment.py +run_name="CTImplyContradict" task=anli +split=dev_r1 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  "task.preprocessor.choices=['Imply', 'Contradict', 'Neither']"

python experiment.py +run_name="CTImplyContradict" task=anli +split=dev_r2 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  "task.preprocessor.choices=['Imply', 'Contradict', 'Neither']"

python experiment.py +run_name="CTImplyContradict" task=anli +split=dev_r3 \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  "task.preprocessor.choices=['Imply', 'Contradict', 'Neither']"

python experiment.py +run_name="CTImplyContradict" task=cb +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  "task.preprocessor.choices=['Imply', 'Contradict', 'Neither']"

python experiment.py +run_name="CTBase" task=craigslist_bargains +split=validation \
  num_proc=4 \
  prompt_experiment_mode=cross_task \
  prompt_path=prompts/general_fixed_choice.yaml \
  model_name="bigscience/T0_3B" \
  evaluation.length_normalization=True \
  batch_size=1
