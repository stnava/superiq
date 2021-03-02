import ants
from superiq.pipeline_utils import *
import pandas
import os
import boto3
import multiprocessing as mp


bucket = "eisai-basalforebrainsuperres2"
metadata = "metadata/full_metadata_20210208.csv"
version = "v01"
prefix = f"superres-pipeline-{version}/ADNI/"
stack_filename = f'stacked_bf_volumes_{version}.csv'
expanded_filename = f'expanded_bf_volumes_{version}.csv'
pivoted_filename = f'pivoted_bf_volumes_{version}.csv'
merge_filename = f"data_w_metadata_{version}.csv"
s3_prefix = "volume_measures/"
s3 = boto3.client('s3')

stack = True
expand = False
pivot =  True
merge = True

def get_files(k):
    bucket = "eisai-basalforebrainsuperres2" # Param
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
    os.remove(path)
    return df

if stack:
    keys = list_images(bucket, prefix)
    keys = [i for i in keys if ".csv" in i]

    with mp.Pool() as p:
        dfs = p.map(get_files, keys)
    #for k in keys:
    #    path = get_s3_object(bucket, k, "tmp")
    #    df = pd.read_csv(path)
    #    fields = ["Label", 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
    #    df = df[fields]
    #    new_rows = []
    #    for i,r in df.iterrows():
    #        label = int(r['Label'])
    #        fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
    #        select_data = r[fields]
    #        values = select_data.values
    #        field_values = zip(fields, values)
    #        for f in field_values:
    #            new_df = {}
    #            new_df['Measure'] = f[0]
    #            new_df['Value'] = f[1]
    #            new_df['Label'] = label
    #            new_df = pd.DataFrame(new_df, index=[0])
    #            new_rows.append(new_df)

    #    df = pd.concat(new_rows)
    #    filename = path.split('/')[-1]
    #    split = filename.split('-')
    #    name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process", "Name"]
    #    zip_list = zip(name_list, split)
    #    for i in zip_list:
    #        df[i[0]] = i[1]
    #    df['OriginalOutput'] = "-".join(split[:5]) + ".nii.gz"
    #    dfs.append(df)
    #    os.remove(path)

    stacked = pd.concat(dfs)
    stacked.to_csv(stack_filename, index=False)


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
    #s3.upload_file(pivoted_filename, bucket, pivoted_filename)

if merge:
    data = pd.read_csv(pivoted_filename)
    meta = get_s3_object(bucket, metadata, "tmp")
    metadf = pd.read_csv(meta)
    os.remove(meta)
    merge = pd.merge(metadf, data, how="right", left_on="filename", right_on="OriginalOutput",suffixes=("","_x"))
    merge.drop(['Repeat_x'], inplace=True, axis=1)
    merge.to_csv(merge_filename, index=False)
    s3.upload_file(merge_filename, bucket, s3_prefix + merge_filename)

