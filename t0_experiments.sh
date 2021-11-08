#!/bin/bash
# Run the experiments
python experiment.py length_normalization=True task=hellaswag +split=validation model_name="bigscience/T0_3B"
python experiment.py force_generation=True task=hellaswag +split=validation model_name="bigscience/T0_3B"
