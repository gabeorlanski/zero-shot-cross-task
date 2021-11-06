#!/bin/bash
# Run the experiments
python experiment.py task=copa model_name="bigscience/T0_3B"
python experiment.py task=anli model_name="bigscience/T0_3B"
python experiment.py task="craigslist_bargains" model_name="bigscience/T0_3B"
python experiment.py task=rte model_name="bigscience/T0_3B"
python experiment.py task=wsc model_name="bigscience/T0_3B"
python experiment.py task=wic model_name="bigscience/T0_3B"
