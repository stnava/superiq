#!/bin/bash

aws_profile=${1:-default}

docker build \
    --build-arg AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile "$aws_profile") \
    --build-arg AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile "$aws_profile") \
    -t superiq_test_image \
    .

docker run --rm -it \
    --name superiq_test \
    superiq_test_image:latest \
    bash 
