#!/bin/bash
# Run the experiments
python experiment.py task=cb +split=validation model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r1 model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r2 model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r3 model_name="bigscience/T0_3B"
python experiment.py task=rte +split=validation model_name="bigscience/T0_3B"
python experiment.py task="wsc.fixed" +split=validation model_name="bigscience/T0_3B"
python experiment.py task=wic +split=validation model_name="bigscience/T0_3B"

python experiment.py length_normalization=True task=cb +split=validation model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=anli +split=dev_r1 model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=anli +split=dev_r2 model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=anli +split=dev_r3 model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=rte +split=validation model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task="wsc.fixed" +split=validation model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=wic +split=validation model_name="bigscience/T0_3B"

python experiment.py task=hellaswag +split=validation model_name="bigscience/T0_3B"
python experiment.py length_normalization=True task=hellaswag +split=validation model_name="bigscience/T0_3B"
python experiment.py force_generation=True task=hellaswag +split=validation model_name="bigscience/T0_3B"
