import os
threads
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads
import os.path
from os import path
import glob as glob


import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import check_for_labels_in_image
from pipeline_utils import *
from superiq import list_to_string
from superiq import basalforebrainSR, basalforebrainOR


def main(input_config):
    config = LoadConfig(input_config)
    input_image = get_pipeline_data(
            "brain_ext-bxtreg_n3.nii.gz",
            config.input_value,
            config.pipeline_bucket,
            config.pipeline_prefix,
    )
    input_image = ants.image_read(input_image) 

    template = get_s3_object(config.template_bucket, config.template_key, "data")
    template = ants.image_read(template) 
    
    templateL = get_s3_object(config.template_bucket, config.template_label_key, "data")
    templateL = ants.image_read(templateL) 

    model_path = get_s3_object(config.model_bucket, config.model_key, "models")
    
    atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
    brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
    brains.sort()
    brains = [ants.image_read(i) for i in brains] 
    
    atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
    brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
    brainsSeg.sort()
    brainsSeg = [ants.image_read(i) for i in brainsSeg] 
     
    havelabels = check_for_labels_in_image( wlab, templateL )
    
    if not havelabels:
        raise Exception("Label missing from the template")
    
    output_filename = "outputs/" + config.output_file_prefix 
    output_filename_sr = output_filename + "_SR.nii.gz"
    output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
    
    bfSR = basalforebrainSR(
            input_image=input_image,
            template=template,
            templateL=templateL,
            model_path=model_path,
            atlas_images=brains,
            atlas_labels=brainsSeg,
            wlab=config.wlab,
            sr_params=config.sr_params,
            seg_params=config.seg_params,
    )
    srseg = bfSR['SR_Img']  
    probseg = bfSR['SR_Seg'] 
    

    get_label_geo(
            probseg,
            srseg, 
            config.process_name,
            config.input_value,
            resolution="SR",
    )
    plot_output(
        srseg, 
        "outputs/basalforebrain-SR_ortho_plot.png",
        probseg,
    )
    handle_outputs(
        config.input_value,
        config.output_bucket,
        config.output_prefix,
        config.process_name,
        env=config.environment,
    )
