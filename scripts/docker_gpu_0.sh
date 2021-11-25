#!/bin/bash
# Run batch 1 in docker
docker run --gpus device=0 --name batch_1 nnsem:latest bash ./scripts/batch_1.sh
docker kill batch_1
docker rm batch_1