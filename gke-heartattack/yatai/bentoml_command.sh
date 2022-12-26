#!/bin/bash

TOKEN=$1
URL=$2

bentoml yatai login --api-token $TOKEN --endpoint $URL
bentoml build
bentoml push iris_classifier:latest
