#!/bin/bash
# Run the experiments
python experiment.py task=cb model_name=$0 disable_tracking=True
python experiment.py task=anli model_name=$0 disable_tracking=True
python experiment.py task=rte model_name=$0 disable_tracking=True
python experiment.py task=wsc model_name=$0 disable_tracking=True
python experiment.py task=wic model_name=$0 disable_tracking=True
