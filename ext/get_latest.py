import boto3
import os


def get_latest():
    Bucket='invicro-data-shared'
    s3 = boto3.client('s3')
    obj = s3.list_objects_v2(
            Bucket=Bucket,
            Prefix='antspy-builds/',
    )['Contents']
    keys = [i['Key'] for i in  obj]
    whl_files = [i for i in keys if '.whl' in i]
    dates = [i.split('/')[-2] for i in whl_files]
    latest = max(dates)
    latest_file = [i for i in whl_files if latest in i][0]
    file_name = "ext/" +latest_file.split('/')[-1]
    print(file_name)
    s3.download_file(
            Bucket,
            latest_file,
            file_name,
    )

    obj = s3.list_objects_v2(
            Bucket=Bucket,
            Prefix='antspynet-builds/',
    )['Contents']
    keys = [i['Key'] for i in  obj]
    whl_files = [i for i in keys if '.whl' in i]
    dates = [i.split('/')[-2] for i in whl_files]
    latest = max(dates)
    latest_file = [i for i in whl_files if latest in i][0]
    file_name = "ext/" +latest_file.split('/')[-1]
    print(latest_file)
    s3.download_file(
            Bucket,
            latest_file,
            file_name
    ) 

if __name__=='__main__':
    get_latest()
