#!/bin/bash
# Run the experiments
python experiment.py task=cb +split=validation num_proc=4 debug=False use_general_prompts=True
python experiment.py task=anli +split=validation num_proc=4 debug=False use_general_prompts=True
