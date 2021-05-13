#!/bin/bash

container_name="$1"
script_path=$2
config_path=$3
antspy_hash="86cd1749ec28e71176780c1dac9dd26b9bb690cd"
antspynet_hash="098d9def7b1b52895596f26050b35e03ab8d0910"
superiq_hash=$(git rev-parse HEAD)
aws_profile=${4:-default}

docker build \
    --build-arg AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id --profile "$aws_profile")\
    --build-arg AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key --profile "$aws_profile")\
    --build-arg antspy_hash=$antspy_hash \
    --build-arg antspynet_hash=$antspynet_hash \
    --build-arg superiq_hash=$superiq_hash \
    -t $container_name .

docker run --rm -it \
    --name $container_name \
    -e cpu_threads="8" \
    $container_name:latest \
    python3 $2 $3
