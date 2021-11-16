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

python experiment.py +run_name="AC" task=cb +split=validation \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r1 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r2 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"

python experiment.py +run_name="AC" task=anli +split=dev_r3 \
  num_proc=4 \
  debug=False \
  use_general_prompts=True \
  model_name="bigscience/T0_3B"
