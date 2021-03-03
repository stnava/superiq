import os
threads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import tensorflow as tf

import antspynet
import math
import ants
import sys
from superiq.pipeline_utils import *
from superiq import deep_dkt


def main(input_config):
    config = LoadConfig(input_config)
    original_image_path = config.input_value

    input_image = get_pipeline_data(
            "brain_ext-bxtreg_n3.nii.gz",
            original_image_path,
            config.pipeline_bucket,
            config.pipeline_prefix,
            "/data"
    )

    input_image = ants.image_read(input_image)

    wlab = config.wlab

    template = antspynet.get_antsxnet_data("biobank")
    template = ants.image_read(template)

    sr_model = get_s3_object(config.model_bucket, config.model_key, "/data")
    #mdl = tf.keras.models.load_model(model_path)

    output_path = "/outputs/"
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

    handle_outputs(
        config.input_value,
        config.output_bucket,
        config.output_prefix,
        config.process_name,
        output_path,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
