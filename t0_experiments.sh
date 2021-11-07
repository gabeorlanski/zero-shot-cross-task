#!/bin/bash
# Run the experiments
python experiment.py task=cb +split=validation model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r1 model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r2 model_name="bigscience/T0_3B"
python experiment.py task=anli +split=dev_r3 model_name="bigscience/T0_3B"
python experiment.py task=rte +split=validation model_name="bigscience/T0_3B"
python experiment.py task=wsc +split=validation model_name="bigscience/T0_3B"
python experiment.py task="wsc.fixed" +split=validation model_name="bigscience/T0_3B"
python experiment.py task=wic +split=validation model_name="bigscience/T0_3B"
