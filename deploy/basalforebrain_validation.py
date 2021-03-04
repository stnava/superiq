import os
# set number of threads - this should be optimized for your compute instance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import pandas as pd
from superiq.pipeline_utils import *
import superiq
import ants
import tensorflow as tf


def basalforebeain_validation(config):
    tdir = "/tmp"
    if ( not path. exists( tdir ) ):
        os.makdirs(tdir)
        #raise RuntimeError('Failed to find the data directory')
    c = LoadConfig(config)
    print("====> Getting remote data")

    target_image_path = get_s3_object(c.input_bucket, c.input_value, tdir)
    target_image = ants.image_read(target_image_path)

    target_image_base = target_image.split('/')[-1].split('.')[0]
    target_image_label_name = c.atlas_label_prefix +  target_image_base + '_JLFSegOR.nii.gz'
    target_image_labels_path = get_s3_object(c.input_bucket, target_image_label_name, tdir)
    target_image_labels = ants.image_read(target_image_labels_path)

    template = get_s3_object(c.template_bucket, c.template_key, tdir)
    template = ants.image_read(template)
    templateL = get_s3_object(c.template_bucket, c.template_label_key, tdir)
    templateL = ants.image_read(templateL)

    model_path = get_s3_object(c.model_bucket, c.model_key, tdir)
    sr_model = tf.keras.model.load_model(model_path)

    atlas_image_keys = list_images(c.atlas_bucket, c.atlas_image_prefix)
    atlas_image_keys = [i for i in atlas_image_keys if i != target_image_path]
    brains = [get_s3_object(c.atlas_bucket, k, tdir) for k in atlas_image_keys]
    brains.sort()

    atlas_label_keys = list_images(c.atlas_bucket, c.atlas_label_prefix)
    atlas_label_keys = [i for i in atlas_label_keys if i != target_image_labels_path]
    brainsSeg = [get_s3_object(c.atlas_bucket, k, tdir) for k in atlas_label_keys]
    brainsSeg.sort()

    seg_params=c.seg_params

    seg_params_sr=c.seg_params_sr

    sr_params=c.sr_params
    wlab = c.wlab

    validation_params = {
        "target_image": target_image,
        "target_labels": target_image_labels,
        "segmentation_numbers": wlab,
        "template": template,
        "template_segmentation": templateL,
        "library_intensity": atlas_images,
        "library_segmentation": atlas_labels,
        "seg_params": seg_params,
        "seg_params_sr": seg_params_sr,
        "sr_params": sr_params,
        "sr_model": model_path,
        "forward_transforms": None,
    }
    output = native_to_superres_ljlf_segmentation
    return test_params
