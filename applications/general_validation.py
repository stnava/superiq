import os
# set number of threads - this should be optimized for your compute instance
mynt="16"
os.environ["TF_NUM_INTEROP_THREADS"] = mynt
os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt

import os.path
from os import path
import glob as glob

import math
import tensorflow
import ants
import antspynet
import tensorflow as tf
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from superiq import super_resolution_segmentation_per_label
from superiq import ljlf_parcellation
from superiq import images_to_list
from superiq import check_for_labels_in_image
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation
from superiq import native_to_superres_ljlf_segmentation
from superiq import list_to_string
from superiq.pipeline_utils import *


# Change this
tdir = "/home/ec2-user/superiq/validation"
if ( not path. exists( tdir ) ):
    raise RuntimeError('Failed to find the data directory')

template_bucket = "invicro-pipeline-inputs"
template_key = "adni_templates/adni_template.nii.gz"
template_label_key = "adni_templates/adni_template_dkt_labels.nii.gz"
model_bucket = "invicro-pipeline-inputs"
model_key = "models/SEGSR_32_ANINN222_3.h5" 
atlas_bucket = "invicro-pipeline-inputs"
atlas_image_prefix = "OASIS30/Brains/"
atlas_label_prefix = "OASIS30/Segmentations"


template = get_s3_object(template_bucket, template_key, tdir)
template = ants.image_read(template)
templateL = get_s3_object(template_bucket, template_label_key, tdir)
templateL = ants.image_read(templateL)

model_path = get_s3_object(model_bucket, model_key, tdir)
mdl = tf.keras.models.load_model(model_path)

atlas_image_keys = list_images(atlas_bucket, atlas_image_prefix)
brains = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_image_keys]
brains.sort()
brains = [ants.image_read(i) for i in brains]

atlas_label_keys = list_images(atlas_bucket, atlas_label_prefix)
brainsSeg = [get_s3_object(atlas_bucket, k, tdir) for k in atlas_label_keys]
brainsSeg.sort()
brainsSeg = [ants.image_read(i) for i in brainsSeg]


# Pull from s3
brains = glob.glob(tdir+"segmentation_libraries/OASIS30/Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"segmentation_libraries/OASIS30/SegmentationsJLFOR/*")
brainsSeg.sort()

# Pull from s3
templatefilename = tdir + "template/adni_template.nii.gz"
templatesegfilename = tdir + "template/adni_template_dkt_labels.nii.gz"

# Will vary by tool
seg_params={
    'submask_dilation': 8,
    'reg_iterations': [100, 100, 20],
    'searcher': 1,
    'radder': 2,
    'syn_sampling': 32,
    'syn_metric': 'mattes',
    'max_lab_plus_one': True, 'verbose': False}

seg_params_sr={
    'submask_dilation': seg_params['submask_dilation']*1,
    'reg_iterations': seg_params['reg_iterations'],
    'searcher': seg_params['searcher'],
    'radder': seg_params['radder'],
    'syn_sampling': seg_params['syn_sampling'],
    'syn_metric': seg_params['syn_metric'],
    'max_lab_plus_one': True, 'verbose': False}

sr_params={"upFactor": [2,2,2], "dilation_amount": seg_params["submask_dilation"], "verbose":True}

# Define labels
#wlab = [36,55,57] # for PPMI
#wlab = [47,116,122,154,170] # eisai cortex
wlab = [75,76] # basal forebrain

# store output data
#brainName = []
#dicevalNativeSeg = []
#dicevalSRNativeSeg = []
#dicevalSRSeg = []
#dicevalSRSeg2 = []
#evalfn='./dkt_eval' + list_to_string( wlab ) + '.csv'
#print( "Labels:" + list_to_string( wlab ) + " " + evalfn, " : n : ", len( brains ) )


native_to_superres_ljlf_segmentation_params = {
    "target_image": "",
    "segmentation_numbers": wlab,
    "template": templatefilename,
    "template_segmenation": templatesegfilename,
    "library_intensity": "",
    "library_segmentation": "",
    "seg_params": seg_params,
    "seg_params_sr": seg_params_sr,
    "sr_params": sr_params,
    "sr_model": mdl,
    "forward_transforms": None,
}



def leave_one_out_cross_validation(
    evaluation_function,
    evaluation_parameters,
    atlas_images,
    atlas_labels,
):
    records = []
    for k in range(len(atlas_images)):
        # Get the name of the target 
        localid=os.path.splitext( os.path.splitext( os.path.basename( atlas_images[k]) )[0])[0]
        print( str(k) + " " + localid)
    
        brainsLocal=atlas_images.copy()
        brainsSegLocal=atlas_labels.copy()
        # Remove the target brain from the atlas set 
        del brainsLocal[k:(k+1)]
        del brainsSegLocal[k:(k+1)]
        original_image = ants.image_read(atlas_images[k])
        evaluation_parameters['target_image'] = original_image 
        evaluation_parameters['library_intensity'] = images_to_list(brainsLocal)
        evaluation_parameters['library_segmentation'] = images_to_list(brainsSegLocal)

        #sloop = native_to_superres_ljlf_segmentation(
        sloop = evaluation_function(**evaluation_parameters)
    
        # first - create a SR version of the image and the ground truth
        # NOTE: we binarize the labels
        # NOTE: the below call would only be used for evaluation ie when we have GT
        nativeGroundTruth = ants.image_read(brainsSeg[k])
        nativeGroundTruth = ants.mask_image( nativeGroundTruth, nativeGroundTruth, level = wlab, binarize=False )
        sr_params = evaluation_parameters['sr_params'] 
        gtSR = super_resolution_segmentation_per_label(
                imgIn = ants.iMath( original_image, "Normalize"),
                segmentation = nativeGroundTruth, # usually, an estimate from a template, not GT
                upFactor = sr_params['upFactor'],
                sr_model = evaluation_paramerters['sr_model'],
                segmentation_numbers = evaluation_parameters['wlab'],
                dilation_amount = sr_params['dilation_amount'],
                verbose = sr_params['verbose']
                )
        nativeGroundTruthProbSR = gtSR['probability_images'][0]
        nativeGroundTruthSR = gtSR['super_resolution_segmentation']
        nativeGroundTruthBinSR = ants.mask_image( nativeGroundTruthSR, nativeGroundTruthSR, wlab, binarize=True)
    
        # The full method involves:  (GT denotes ground truth)
        # [0.0] use template-based mapping to estimate initial labels
        # [1.0] run LJLF at native resolution (evaluate this wrt native res GT)
        #   [1.1] evaluate [1.0] wrt NN-Up-GT
        # [2.0] perform local simultaneous SR-Image and SR-Seg based on output of [1.0] (evaluate this wrt SR-GT)
        #   [2.1] evaluate [2.0] wrt NN-Up-GT
        # [3.0] run LJLF at SR based on [2.0] (evaluate this at SR wrt SR-GT)
        #   [3.1] evaluate [3.0] this wrt NN-Up-GT
        mypt = 0.5
        srsegLJLF = ants.threshold_image( sloop['srSeg']['probsum'], mypt, math.inf )
        nativeOverlapSloop = ants.label_overlap_measures( nativeGroundTruth, sloop['nativeSeg']['segmentation'] )
        srOnNativeOverlapSloop = ants.label_overlap_measures( nativeGroundTruthSR, sloop['srOnNativeSeg']['super_resolution_segmentation'] )
        srOverlapSloop = ants.label_overlap_measures( nativeGroundTruthSR, sloop['srSeg']['segmentation'] )
        srOverlap2 = ants.label_overlap_measures( nativeGroundTruthBinSR, srsegLJLF )
        # collect the 3 evaluation results - ready for data frame
        brainName = []
        dicevalNativeSeg = []
        dicevalSRNativeSeg = []
        dicevalSRSeg = []
        dicevalSRSeg2 = []

        brainName.append( localid )
        dicevalNativeSeg.append(nativeOverlapSloop["MeanOverlap"][0])
        dicevalSRNativeSeg.append( srOnNativeOverlapSloop["MeanOverlap"][0])
        dicevalSRSeg.append( srOverlapSloop["MeanOverlap"][0])
        dicevalSRSeg2.append( srOverlap2["MeanOverlap"][0])
        print( brainName[k] + ": N: " + str(dicevalNativeSeg[k]) + " SRN: " +  str(dicevalSRNativeSeg[k])+ " SRN: " +  str(dicevalSRSeg[k]) )
        ################################################################################
        dict = {
            'name': brainName,
            'diceNativeSeg': dicevalNativeSeg,
            'diceSRNativeSeg': dicevalSRNativeSeg,
            'diceSRSeg': dicevalSRSeg }
        df = pd.DataFrame(dict)
        evalfn='./dkt_eval' + list_to_string( wlab ) + '.csv'
        df.to_csv( evalfn )
        break 
        ################################################################################

# these are the outputs you would write out, along with label geometry for each segmentation
#ants.image_write( sloop['srOnNativeSeg']['super_resolution'], '/tmp/tempI.nii.gz' )
#ants.image_write( nativeGroundTruthSR, '/tmp/tempGT.nii.gz' )
#ants.image_write( sloop['srSeg']['segmentation'], '/tmp/tempSRSeg.nii.gz' )
#ants.image_write( sloop['nativeSeg']['segmentation'], '/tmp/tempORSeg.nii.gz' )
