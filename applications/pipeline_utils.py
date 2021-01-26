import os
import pandas as pd
import pickle
import boto3
import json
import sys
import ast
import glob

def get_s3_object(bucket, key, local_dir):
    s3 = boto3.client('s3')
    basename = key.split('/')[-1]
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    local = f'{local_dir}/{basename}'
    s3.download_file(
            bucket,
            key,
            local, 
    )
    return local


class LoadConfig:
    """
    Creates a config object for referencing variables in the passed
    config file.

    Arguments
    ---------
    config : string
        A string that can be parsed into a dict via ast.literal_eval or a
        string representing the file path to the config json file
    """
    def __init__(self, config):
        try:
            # Parse a json string into a python dict (AWS BATCH)
            params = ast.literal_eval(config)
        except ValueError:
            # If using actual json file
            with open(config, 'r') as f:
                data = f.read()
            params = json.loads(data)

        parameters = params['parameters'] 
        for key in parameters:
            setattr(self, key, parameters[key])

        if params['aws_profile'] != "role":
            os.environ['AWS_PROFILE'] = params['aws_profile']
        print(f"AWS profile set to {params['aws_profile']}")
    
    def __repr__(self):
        return f"config: {self.__dict__}"


def handle_outputs(input_path, output_bucket, output_prefix, process_name, dev=False):
    outputs = [i for i in os.listdir('outputs')]
    path, basename = derive_s3_path(input_path)
    prefix = output_prefix + process_name + '/' + path
    for output in outputs:
        filename = output.split('/')[-1]
        obj_name = basename + '-' + '-' + process_name + '-' + filename
        obj_path = prefix + obj_name
        print(f"{output} -> {obj_path}") 
        if not dev:
            s3.upload_file(
                    output,
                    output_bucket,
                    obj_path,
            )


def get_pipeline_data(filename, initial_image_key, bucket, prefix):
    path, _ = derive_s3_path(initial_image_key)  
    print(path) 
    key_list = list_images(bucket, prefix + path)  
    key = [i for i in key_list if i.endswith(filename)]
    if len(key) != 1:
        raise ValueError(f'{len(key)} objects were found with that suffix')
    else:
        key = key[0]
        local = get_s3_object(bucket, key, "data")
        return local


def get_library(local_path, bucket, prefix):
    s3 = boto3.client('s3')
    keys = list_images(bucket, prefix)
    for k in keys:
        basename = k.split('/')[-1]
        local = local_path + basename
        s3.download_file(
            bucket,
            k,
            local
        )


def derive_s3_path(image_path):
    basename = image_path.split('/')[-1].replace('.nii.gz', '')
    loc = basename.split('-')
    path = '/'.join(loc) + '/'
    return path, basename


def list_images(bucket, prefix):
    s3 = boto3.client('s3')
    items = []
    kwargs = {
        'Bucket': bucket,
        'Prefix': prefix,
    }
    while True:
        objects = s3.list_objects_v2(**kwargs)
        try: 
            for obj in objects['Contents']:
                key =  obj['Key']
                items.append(key)
        except KeyError:
            raise KeyError("No keys found under prefix {prefix}")
        try:
            kwargs['ContinuationToken'] = objects['NextContinuationToken']
        except KeyError:
            break
    images = [i for i in items if "." in i]
    return images


def get_label_geo(
        labeled_image, # The ants image to be labeled 
        initial_image, # The unlabeled image, used in the label stats
        process, # Name of the process
        input_key, # Replace
        label_map_params=None, # Ignore for now
        resolution='OR',
        direction=None):
    import ants
    print('Starting Label Geometry Measures') 
    lgms = ants.label_geometry_measures(
            labeled_image,
    )
    if label_map_params is not None:
        label_map_local = get_input_image(label_map_params['bucket'], label_map_params['key'])
        label_map = pd.read_csv(label_map_local)
        label_map.columns = ['LabelNumber', 'LabelName']
        label_map.set_index('LabelNumber', inplace=True)
        label_map_dict = label_map.to_dict('index')
    
    new_rows = []
    for i,r in lgms.iterrows():
        label = int(r['Label'])
        if label_map_params is not None:
            label = label_map_dict[label]['LabelName']
            clean_label = label.replace(' ', '_')
        else:
            clean_label = label
        fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared'] 
        select_data = r[fields] 
        values = select_data.values
        field_values = zip(fields, values)
        
        for f in field_values:
            new_df = {}
            new_df['Measure'] = f[0]
            new_df['Value'] = f[1]
            new_df['Process'] = process 
            new_df['Resolution'] = resolution
            if direction is not None: 
                new_df['Side'] = direction
            else:
                new_df['Side'] = 'full'
            new_df['Label'] = clean_label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)
    label_stats = ants.label_stats(initial_image, labeled_image)
    label_stats = label_stats[label_stats['Mass']>0]

    for i,r in label_stats.iterrows():
        label = int(r['LabelValue'])
        if label_map_params is not None:
            label = label_map_dict[label]['LabelName']
            clean_label = label.replace(' ', '_') 
        else:
            clean_label = label
        fields = ['Mean'] 
        select_data = r[fields]  
        values = select_data.values
        field_values = zip(fields, values)
        for f in field_values:
            new_df = {}
            new_df['Measure'] = 'MeanIntensity'
            new_df['Value'] = f[1]
            new_df['Process'] = process 
            new_df['Resolution'] = resolution
            if direction is not None: 
                new_df['Side'] = direction
            else:
                new_df['Side'] = 'full'
            new_df['Label'] = clean_label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)
    label_data = pd.concat(new_rows) 
    
    s3_path, _ = derive_s3_path(input_key)
    split = s3_path.split('/')  

    label_data['Study'] = split[0]
    label_data['Subject'] = split[1]
    label_data['Date'] = split[2]
    label_data['Modality'] =  split[3]
    label_data['Repeat'] = split[4]
    full = label_data
    if direction is None:
        side = "full"
    else:
        side = direction
    output_name = f'outputs/{resolution}-{side}-lgm.csv'
    full.to_csv(output_name, index=False)


def plot_output(img, output_path, overlay=None):
    import ants
    if overlay is None:
        plot = ants.plot_ortho(
                ants.crop_image(img), 
                flat=True, 
                filename=output_path,
        )
    else:
        plot = ants.plot_ortho(
                ants.crop_image(img, overlay), 
                overlay=ants.crop_image(overlay, overlay),
                flat=True,
                filename=output_path,
        )


def dev_output(image, filename):
    """ Output an ants image object to s3 for dev testing """
    import ants
    s3 = boto3.client('s3')
    tmp_file = f'/tmp/{filename}.nii.gz'
    ants.image_write(image, tmp_file)
    output_bucket = 'invicro-test-outputs'
    output_key = f'bf/{filename}' 
    s3.upload_file(tmp_file, output_bucket, output_key)
