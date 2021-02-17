## Deploy to AWS Batch

This folder contains the script used to updated an AWS Batch Job Definition, 
deploy_job.sh. The script should be invoked from the top folder of this 
repositiory.

`./deploy/deploy_job.sh \
    <container_name> \
    <cpu_count> \
    <memory_in_gb> \
    <deployment_script>
