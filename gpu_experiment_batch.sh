#!/bin/bash
# Run the experiments

python experiment.py +run_name="LowerCase" task=cb +split=validation \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=anli +split=dev_r1 \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=anli +split=dev_r2  \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=anli +split=dev_r3  \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=rte +split=validation  \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=wsc +split=validation \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task="wsc.fixed" +split=validation  \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True
python experiment.py +run_name="LowerCase" task=wic +split=validation  \
  model_name="bigscience/T0_3B" \
  evaluation.lowercase_choices=True





