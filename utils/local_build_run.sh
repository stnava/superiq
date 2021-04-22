#!/bin/bash

container_name="$1"
script_path=$2
antspy_hash="e7e8644857a78c442aff5e688ccd491164746b24"
antspynet_hash="b991b14edc7c0aad79fec2cd02afedee49a9c18a"
superiq_hash=$(git rev-parse HEAD)
aws_profile=$3

docker build \
    --build-arg AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile "$aws_profile")\
    --build-arg AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile "$aws_profile")\
    --build-arg antspy_hash=$antspy_hash \
    --build-arg antspynet_hash=$antspynet_hash \
    --build-arg superiq_hash=$superiq_hash \
    -t $container_name .

docker run --rm -it \
    --name $container_name \
    $container_name:latest

