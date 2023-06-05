#!/usr/bin/env bash

mkdir -p mount 
touch ./mount/config.yaml
docker-compose -f docker-compose-dependencies.yaml up -d
docker-compose up 
