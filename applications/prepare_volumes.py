import ants
from superiq.pipeline_utils import *
import pandas
import os
import boto3

bucket = "eisai-basalforebrainsuperres2"
metadata = "metadata/full_metadata.csv"
prefix = "superres-pipeline-20210202/ADNI/"
stack_filename = 'stacked_bf_volumes.csv'
expanded_filename = 'expanded_bf_volumes.csv'
pivoted_filename = 'pivoted_bf_volumes.csv'
s3 = boto3.client('s3')

stack = False
expand = False
pivot =  True
merge = True


if stack:
    keys = list_images(bucket, prefix)
    
    keys = [i for i in keys if ".csv" in i]
    
    dfs = []
    for k in keys:
        print(k)
        path = get_s3_object(bucket, k, "tmp") 
        df = pd.read_csv(path)
        fields = ["Label", 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared'] 
        df = df[fields]      
        new_rows = []
        for i,r in df.iterrows():
            label = int(r['Label'])
            fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared'] 
            select_data = r[fields] 
            values = select_data.values
            field_values = zip(fields, values)
            for f in field_values:
                new_df = {}
                new_df['Measure'] = f[0]
                new_df['Value'] = f[1]
                new_df['Label'] = label
                new_df = pd.DataFrame(new_df, index=[0])
                new_rows.append(new_df)

        df = pd.concat(new_rows)
        filename = path.split('/')[-1]
        split = filename.split('-')
        name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process", "Name"]
        zip_list = zip(name_list, split)
        for i in zip_list:
            df[i[0]] = i[1]
        df['OriginalOutput'] = "-".join(split[:5]) + ".nii.gz"
        dfs.append(df)
        os.remove(path) 
    
    stacked = pd.concat(dfs)
    stacked.to_csv(stack_filename, index=False)
    s3.upload_file(stack_filename, bucket, stack_filename)

if expand:
    lgms = pd.read_csv(stack_filename)
    lgms['Name'] = [i.split('.')[0] for i in lgms['Name']]
    new_rows = []
    for i,r in lgms.iterrows():
        label = int(r['Label'])
        fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared'] 
        select_data = r[fields] 
        values = select_data.values
        field_values = zip(fields, values)
        for f in field_values:
            new_df = {}
            new_df['Measure'] = f[0]
            new_df['Value'] = f[1]
            new_df['Label'] = label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)
    new_df = pd.concat(new_rows)
    new_df.to_csv(expanded_filename, index=False)
    s3.upload_file(expanded_filename, bucket, expanded_filename)

if pivot:
    df = pd.read_csv(stack_filename)
    df['Name'] = [i.split('.')[0] for i in df['Name']]
    pivoted = df.pivot(
        index=['Project','Subject','Date', 'Modality', 'Repeat', "OriginalOutput"],
        columns=['Measure', 'Label', 'Process','Name']) 
    
    columns = []
    for c in pivoted.columns:
        cols = [str(i) for i in c]
        column_name = '-'.join(cols[1:])
        columns.append(column_name)
    
    pivoted.columns = columns
    pivoted.reset_index(inplace=True)
    final_csv = pivoted 

    final_csv['Repeat'] = [str(i).zfill(3) for i in final_csv['Repeat']]
    final_csv.to_csv(pivoted_filename, index=False) 
    s3.upload_file(pivoted_filename, bucket, pivoted_filename)    

if merge:
    data = pd.read_csv(pivoted_filename) 
    meta = get_s3_object(bucket, metadata, "tmp")
    metadf = pd.read_csv(meta)
    os.remove(meta) 
    merge = pd.merge(metadf, data, how="right", left_on="new_filename", right_on="OriginalOutput")
    merge_filename = "data_w_metadata.csv"
    merge.to_csv(merge_filename, index=False)
    s3.upload_file(merge_filename, bucket, merge_filename)

