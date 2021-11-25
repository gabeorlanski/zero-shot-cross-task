#!/bin/bash
# Run batch 2 in docker
docker run --gpus device=0 --name batch_2 nnsem:latest bash scripts/batch_2.sh
docker kill batch_2
docker rm batch_2