container_name="$1"
cpus=$2
memory_in_gb="$(($3 * 1000))"
script_path=$4
antspy_hash="86cd1749ec28e71176780c1dac9dd26b9bb690cd"
antspynet_hash="1da45624dcc37966d7571df5e175fa46d27509fb"
superiq_hash=$(git rev-parse HEAD)

repo="651875258113.dkr.ecr.us-east-1.amazonaws.com"
repo_image="${repo}/${container_name}"
repo_image_latest="${repo}/${container_name}:latest"
job_definition_name=$container_name
job_role="ia-general-s3"

aws ecr get-login-password | docker login --username AWS --password-stdin $repo

docker build \
    --build-arg antspy_hash=$antspy_hash \
    --build-arg antspynet_hash=$antspynet_hash \
    --build-arg superiq_hash=$superiq_hash \
    -t $container_name .


docker tag $container_name $repo_image && \
    docker push $repo_image && \
    aws batch register-job-definition \
    --job-definition-name $job_definition_name \
    --type container \
    --timeout attemptDurationSeconds=36000 \
    --retry-strategy attempts=3 \
    --container-properties \
    '{
        "image": "'"${repo_image_latest}"'",
        "vcpus": '"${cpus}"',
        "memory": '"${memory_in_gb}"',
        "command": [ "python", "'"${script_path}"'", "Ref::config" ],
        "jobRoleArn": "arn:aws:iam::651875258113:role/'"${job_role}"'",
        "volumes": [],
        "environment": [
            {"name": "antspy_hash", "value": "'"${antspy_hash}"'"},
            {"name": "antspynet_hash", "value": "'"${antspynet_hash}"'"},
            {"name": "superiq_hash", "value": "'"${superiq_hash}"'"},
            {"name": "cpu_threads", "value": "'"${cpus}"'"}
        ],
        "mountPoints": [],
        "ulimits": [],
        "resourceRequirements": []
    }' > /dev/null


