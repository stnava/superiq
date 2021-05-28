import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import antspynet
import math
import ants
import sys
#from superiq.pipeline_utils import *
from superiq import deep_dkt, super_resolution_segmentation_per_label
import ia_batch_utils as batch
import pandas as pd


def main(input_config):
    config = batch.LoadConfig(input_config)
    c = config
    if config.environment == 'prod':
        input_image = batch.get_s3_object(
                config.input_bucket,
                config.input_value,
                "data"
        )
        input_image = ants.image_read(input_image)
    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

    wlab = config.wlab

    template = antspynet.get_antsxnet_data("biobank")
    template = ants.image_read(template)
    template =  template * antspynet.brain_extraction(template, 't1')

    sr_model = batch.get_s3_object(config.model_bucket, config.model_key, "data")
    mdl = tf.keras.models.load_model(sr_model)

    output_path = config.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_params={
        "target_image": input_image,
        "segmentation_numbers": wlab,
        "template": template,
        "sr_model": sr_model,
        "sr_params": config.sr_params,
        "output_path": output_path,
    }

    output = deep_dkt(**input_params)

    label_or = output['labels_or']
    label_sr = output['labels_sr']

    labels_or = pd.read_csv(label_or)
    labels_sr = pd.read_csv(label_sr)

    vals = {
        "OR": labels_or,
        "SR": labels_sr,
    }

    for key, value in vals.items():
        split = config.input_value.split('/')[-1].split('-')
        rec = {}
        rec['originalimage'] = "-".join(split[:5]) + '.nii.gz'
        rec['hashfields'] = ['originalimage', 'process', 'batchid', 'data']
        rec['batchid'] = c.batch_id
        rec['project'] = split[0]
        rec['subject'] = split[1]
        rec['date'] = split[2]
        rec['modality'] = split[3]
        rec['repeat'] = split[4]
        rec['process'] = 'deep_dkt'
        rec['name'] = "deep_dkt"
        rec['version'] = config.version
        rec['extension'] = ".nii.gz"
        rec['resolution'] = key
        df = value[['Label', 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']]
        volumes = df.to_dict('records')
        for r in volumes:
            label = r['Label']
            r.pop("Label", None)
            for k, v in r.items():
                rec['data'] = {}
                rec['data']['label'] = label
                rec['data']['key'] = k
                rec['data']['value'] = v
                print(rec)
                batch.write_to_dynamo(rec)


    if config.environment == 'prod':
        batch.handle_outputs(
            config.output_bucket,
            config.output_prefix,
            config.input_value,
            config.process_name,
            config.version
        )

    else:
        raise ValueError(f"The environemnt {config.environment} is not recognized")

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
