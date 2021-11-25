#!/bin/bash
# Helper script for pulling and building the docker image
git pull
docker build -t nlpproject:latest .