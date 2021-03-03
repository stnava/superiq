import os
# set number of threads - this should be optimized for your compute instance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import pandas as pd
from superiq.pipeline_utils import *
import superiq

def create_test_params(config):
    tdir = "/tmp"
    if ( not path. exists( tdir ) ):
        os.makdirs(tdir)
        #raise RuntimeError('Failed to find the data directory')
    c = LoadConfig(config)
    print("====> Getting remote data")

    template = get_s3_object(c.template_bucket, c.template_key, tdir)
    templateL = get_s3_object(c.template_bucket, c.template_label_key, tdir)
    model_path = get_s3_object(c.model_bucket, c.model_key, tdir)

    atlas_image_keys = list_images(c.atlas_bucket, c.atlas_image_prefix)
    brains = [get_s3_object(c.atlas_bucket, k, tdir) for k in atlas_image_keys]
    brains.sort()

    atlas_label_keys = list_images(c.atlas_bucket, c.atlas_label_prefix)
    brainsSeg = [get_s3_object(c.atlas_bucket, k, tdir) for k in atlas_label_keys]
    brainsSeg.sort()

    seg_params=c.seg_params

    seg_params_sr=c.seg_params_sr

    sr_params=c.sr_params
    wlab = c.wlab

    test_params = {
        "target_image": "",
        "segmentation_numbers": wlab,
        "template": template,
        "template_segmentation": templateL,
        "library_intensity": "",
        "library_segmentation": "",
        "seg_params": seg_params,
        "seg_params_sr": seg_params_sr,
        "sr_params": sr_params,
        "sr_model": model_path,
        "forward_transforms": None,
    }
    return test_params
