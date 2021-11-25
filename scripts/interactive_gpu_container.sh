#!/bin/bash
# Open an interactive docker container, run it, then close it
docker create --gpus device=$0 -it --name test-container nlpproject
docker start test-container
docker exec -it test-container bash
docker kill test-container
docker rm test-container