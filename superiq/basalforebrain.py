import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"
import os.path
from os import path
import glob as glob

import tensorflow
import ants
import sys
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd

from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import check_for_labels_in_image
from superiq.pipeline_utils import * 
from superiq import list_to_string

def basalforebrain(
        config=None,
        templatefilename=None,
        templatesegfilename=None,
        infn=None,
        model_file_name=None,
        atlas_image_dir=None,
        atlas_label_dir=None,
        sr_params={"upFactor": [2,2,2], "dilation_amount": 12, "verbose":True},
        seg_params={
            "wlab":[75,76], "submask_dilation":20, "reg_iteration": [100,50,10],
            "searcher": 2, "radder": 3, "syn_sampling": 2, "syn_metric": "CC",
            "max_lab_plus_one": False, "verbose": True
        },
        env="prod"): 
     
    if config: 
        config = LoadConfig(config)
        output_filename = "outputs/" + config.output_name
        # TODO  
        templatefilename = get_s3_object(config.template_bucket, config.template_key, "data")
        # TODO 
        templatesegfilename = get_s3_object(config.template_bucket, config.template_label_key, "data")
        # TODO 
        infn = get_pipeline_data(
                "brain_ext-bxtreg_n3.nii.gz",
                config.input_value,
                config.pipeline_bucket,
                config.pipeline_prefix,
        )
        # TODO 
        model_file_name = get_s3_object(config.model_bucket, config.model_key, "models")
        # TODO
        sr_params = config.sr_params
        seg_params = config.seg_params
   
        # TODO
        if seg_params['wlab']['range']:
            wlab = range(seg_params['wlab']['values'][0],seg_params['wlab']['values'][1])
        else:
            wlab = seg_params['wlab']['values']
        
        # TODO 
        atlas_image_keys = list_images(config.atlas_bucket, config.atlas_image_prefix)
        brains = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_image_keys]
        brains.sort()
        # TODO 
        atlas_label_keys = list_images(config.atlas_bucket, config.atlas_label_prefix)
        brainsSeg = [get_s3_object(config.atlas_bucket, k, "atlas") for k in atlas_label_keys]
        brainsSeg.sort()
   
    else:
        brains = glob.glob(atlas_images_dir+"/*")
        brains.sort()
        brains = brains[0:8] # shorten this for the test application
        brainsSeg = glob.glob(atlas_labels_dir+"/*")
        brainsSeg.sort()
        brainsSeg = brainsSeg[0:8] # shorten this for this test applicationa
        wlab = seg_params['wlab'] 
        output_filename = "outputs/basalforebrain"

    # input data
    imgIn = ants.image_read( infn )
    template = ants.image_read( templatefilename )
    templateL = ants.image_read( templatesegfilename )
    mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this
    
    havelabels = check_for_labels_in_image( wlab, templateL )
    
    if not havelabels:
        raise Exception("Label missing from the template")
    
    # expected output data
    output_filename_sr = output_filename + "_SR.nii.gz"
    output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
    output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
    output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
    
    
    # first, run registration - then do SR in the local region
    if not 'reg' in locals():
        reg = ants.registration( imgIn, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( imgIn, templateL,
              forward_transforms, interpolator="genericLabel" )
    
    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = wlab,
        dilation_amount = sr_params['dilation_amount'],
        verbose = sr_params['verbose']
    )
    
    # write
    initlab0 = ants.apply_transforms( srseg['super_resolution'], templateL,
        forward_transforms, interpolator="genericLabel" )
    ants.image_write( srseg['super_resolution'] , output_filename_sr )
    ants.image_write( initlab0 , output_filename_sr_seg_init )
    
    locseg = ljlf_parcellation(
            srseg['super_resolution'],
            segmentation_numbers=wlab,
            forward_transforms=forward_transforms,
            template=template,
            templateLabels=templateL,
            library_intensity = brains,
            library_segmentation = brainsSeg,
            submask_dilation=seg_params['submask_dilation'],  # a parameter that should be explored
            searcher=seg_params['searcher'],  # double this for SR
            radder=seg_params['radder'],  # double this for SR
            reg_iterations=seg_params['reg_iterations'], # fast test
            syn_sampling=seg_params['syn_sampling'],
            syn_metric=seg_params['syn_metric'],
            max_lab_plus_one=seg_params['max_lab_plus_one'],
            output_prefix=output_filename,
            verbose=seg_params['verbose'],
        )
    probs = locseg['ljlf']['ljlf']['probabilityimages']
    probability_labels = locseg['ljlf']['ljlf']['segmentation_numbers']
    # find proper labels
    whichprob75 = probability_labels.index(wlab[0])
    whichprob76 = probability_labels.index(wlab[1])
    probseg = ants.threshold_image(
      ants.resample_image_to_target(probs[whichprob75], srseg['super_resolution'] ) +
      ants.resample_image_to_target(probs[whichprob76], srseg['super_resolution'] ),
      0.3, 1.0 )
    ants.image_write( probseg,  output_filename_sr_seg )
   
    if config:
        get_label_geo(
                probseg,
                srseg['super_resolution'],
                config.process_name,
                config.input_value,
                resolution="SR",
        )
        plot_output(
            srseg['super_resolution'],
            "outputs/basalforebrain-SR_ortho_plot.png",
            probseg,
        )
        handle_outputs(
            config.input_value,
            config.output_bucket,
            config.output_prefix,
            config.process_name,
            env=env,
        )
        

if __name__ == "__main__":
    basalforebrain(config=sys.argv[1], env='prod')
