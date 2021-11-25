#!/bin/bash
# Open an interactive docker container, run it, then close it
docker create --gpus device=1 -it --name nlpproject-container nlpproject
docker start nlpproject-container
docker exec -it nlpproject-container bash
docker kill nlpproject-container
docker rm nlpproject-container