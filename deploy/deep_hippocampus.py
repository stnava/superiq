import os
threads = "16"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
from superiq.pipeline_utils import *
from superiq import deep_hippo
import sys
import antspynet
import numpy as np

def deep_hippo_deploy(input_config):
    c = LoadConfig(input_config)
    if c.environment == "prod":
        original_image_path = c.input_value
        input_image_path = get_pipeline_data(
            "bxtreg_n3.nii.gz",
            original_image_path,
            c.pipeline_bucket,
            c.pipeline_prefix,
            "data"
        )
        img = ants.image_read(input_image_path)
        template = antspynet.get_antsxnet_data( "biobank" )
        template = ants.image_read( template )
        template = template * antspynet.brain_extraction( template )

        sr_model_path = get_s3_object(
            c.model_bucket,
            c.model_key,
            "data"
        )

        if not os.path.exists(c.local_output_path):
            os.makedirs(c.local_output_path)

        input_params = {
            "img": img,
            "template": template,
            "sr_model_path": sr_model_path,
            "output_path": c.local_output_path,
        }

        outputs = deep_hippo(**input_params)

    if c.environment == "prod":
        handle_outputs(
            c.input_value,
            c.output_bucket,
            c.output_prefix,
            c.process_name,
            c.local_output_path
        )

if __name__ == "__main__":
    config = sys.argv[1]
    deep_hippo_deploy(config)
