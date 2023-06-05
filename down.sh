#!/usr/bin/env bash

docker-compose down
docker-compose -f docker-compose-dependencies.yaml down
