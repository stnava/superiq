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
from superiq import check_for_labels_in_image
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation

def images_to_list( x ):
    outlist = []
    for k in range(len(x)):
        outlist.append( ants.image_read( x[k] ) )
    return outlist


def nativeToSuperResSeg(
    target_image,
    segmentation_numbers,
    template,
    template_segmentation,
    library_intensity,
    library_segmentation,
    seg_params,
    sr_params,
    sr_model,
    forward_transforms=None
     ):

    seg_params_sr={
        'submask_dilation': seg_params['submask_dilation']*1,
        'reg_iterations': seg_params['reg_iterations'],
        'searcher': seg_params['searcher'],
        'radder': seg_params['radder'],
        'syn_sampling': seg_params['syn_sampling'],
        'syn_metric': seg_params['syn_metric'],
        'max_lab_plus_one': True, 'verbose': False}

    if forward_transforms is None:
        print("Registration")
        reg = ants.registration( target_image, template, 'SyN' )
        forward_transforms = reg['fwdtransforms']
        initlab0 = ants.apply_transforms( target_image, template_segmentation,
              forward_transforms, interpolator="nearestNeighbor" )
    else:
        initlab0 = ants.apply_transforms( target_image, template_segmentation,
              forward_transforms, interpolator="nearestNeighbor" )

    # algorithm 1: native resolution LJLF
    nativeseg = basalforebrain_segmentation(
            target_image=ants.iMath(target_image,"Normalize"),
            segmentation_numbers = segmentation_numbers,
            template = template,
            template_segmentation = template_segmentation,
            library_intensity=library_intensity,
            library_segmentation=library_segmentation,
            seg_params = seg_params,
            forward_transforms = forward_transforms
            )

    # algorithm 2: SR on native resolution LJLF
    srOnNativeSeg = super_resolution_segmentation_per_label(
            imgIn = ants.iMath(target_image,"Normalize"),
            segmentation = nativeseg['probseg'],
            upFactor = sr_params['upFactor'],
            sr_model = sr_model,
            segmentation_numbers = [1],
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose'] )
    # the above gives a result that itself can be evaluated
    # algorithm 3: super resolution LJLF
    srseg = basalforebrain_segmentation(
            target_image=srOnNativeSeg['super_resolution'],
            segmentation_numbers = segmentation_numbers,
            template = template,
            template_segmentation = template_segmentation,
            library_intensity=library_intensity,
            library_segmentation=library_segmentation,
            seg_params = seg_params_sr,
            forward_transforms = forward_transforms
            )
    return {
    'nativeSeg':nativeseg,
    'srOnNativeSeg':srOnNativeSeg,
    'srSeg':srseg,
    'forward_transforms':forward_transforms
    }

# get data from here https://ndownloader.figshare.com/files/26224727
tdir = "/Users/stnava/data/superiq_data_resources/"
if ( not path. exists( tdir ) ):
	raise RuntimeError('Failed to find the data directory')

brains = glob.glob(tdir+"segmentation_libraries/OASIS30/Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"segmentation_libraries/OASIS30/Segmentations/*")
brainsSeg.sort()
templatefilename = tdir + "template/adni_template.nii.gz"
templatesegfilename = tdir + "template/adni_template_dkt_labels.nii.gz"

seg_params={
    'submask_dilation': 8,
    'reg_iterations': [100, 100, 20],
    'searcher': 1,
    'radder': 2,
    'syn_sampling': 32,
    'syn_metric': 'mattes',
    'max_lab_plus_one': True, 'verbose': True}

seg_params_sr={
    'submask_dilation': seg_params['submask_dilation']*1,
    'reg_iterations': seg_params['reg_iterations'],
    'searcher': seg_params['searcher'],
    'radder': seg_params['radder'],
    'syn_sampling': seg_params['syn_sampling'],
    'syn_metric': seg_params['syn_metric'],
    'max_lab_plus_one': True, 'verbose': True}

sr_params={"upFactor": [2,2,2], "dilation_amount": seg_params["submask_dilation"], "verbose":True}
mdl = tf.keras.models.load_model("models/SEGSR_32_ANINN222_3.h5")

# store output data
brainName = []
# the three types of output which we will compute in series
dicevalNativeSeg = []
dicevalSRNativeSeg = []
dicevalSRSeg = []
dicevalSRSeg2 = []

for k in range( len(brainName), len( brains ) ):
    localid=os.path.splitext( os.path.splitext( os.path.basename( brains[k]) )[0])[0]
    print( str(k) + " " + localid)
    brainsLocal=brains.copy()
    brainsSegLocal=brainsSeg.copy()
    del brainsLocal[k:(k+1)]
    del brainsSegLocal[k:(k+1)]
    wlab = [75,76]

    sloop = nativeToSuperResSeg(
        target_image = ants.image_read(brains[k]),
        segmentation_numbers = wlab,
        template = ants.image_read(templatefilename),
        template_segmentation = ants.image_read(templatesegfilename),
        library_intensity=images_to_list(brainsLocal),
        library_segmentation=images_to_list(brainsSegLocal),
        seg_params = seg_params,
        sr_params = sr_params,
        sr_model = mdl )

    # first - create a SR version of the image and the ground truth
    # NOTE: we binarize the labels
    # NOTE: the below call would only be used for evaluation ie when we have GT
    nativeGroundTruth = ants.image_read(brainsSeg[k])
    nativeGroundTruthBin = ants.mask_image( nativeGroundTruth, nativeGroundTruth, level = wlab, binarize=True )
    gtSR = super_resolution_segmentation_per_label(
            imgIn = ants.image_read(brains[k]).iMath( "Normalize"),
            segmentation = nativeGroundTruthBin, # usually, an estimate from a template, not GT
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = [1],
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose']
            )
    nativeGroundTruthProbSR = gtSR['probability_images'][0]
    nativeGroundTruthBinSR = ants.threshold_image( nativeGroundTruthProbSR, 0.5, 1.0 )

    # The full method involves:  (GT denotes ground truth)
    # [0.0] use template-based mapping to estimate initial labels
    # [1.0] run LJLF at native resolution (evaluate this wrt native res GT)
    #   [1.1] evaluate [1.0] wrt NN-Up-GT
    # [2.0] perform local simultaneous SR-Image and SR-Seg based on output of [1.0] (evaluate this wrt SR-GT)
    #   [2.1] evaluate [2.0] wrt NN-Up-GT
    # [3.0] run LJLF at SR based on [2.0] (evaluate this at SR wrt SR-GT)
    #   [3.1] evaluate [3.0] this wrt NN-Up-GT
    srsegLJLF = ants.threshold_image( sloop['srSeg']['probsum'], mypt, 2.0 )
    nativeOverlapSloop = ants.label_overlap_measures( nativeGroundTruthBin, sloop['nativeSeg']['probseg'] )
    srOnNativeOverlapSloop = ants.label_overlap_measures( gtSR['super_resolution_segmentation'], sloop['srOnNativeSeg']['super_resolution_segmentation'] )
    srOverlapSloop = ants.label_overlap_measures( gtSR['super_resolution_segmentation'], sloop['srSeg']['probseg'] )
    srOverlap2 = ants.label_overlap_measures( gtSR['super_resolution_segmentation'], srsegLJLF )
    # collect the 3 evaluation results - ready for data frame
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
    df.to_csv('./bf_sr_eval.csv')
    ################################################################################

# these are the outputs you would write out, along with label geometry for each segmentation
ants.image_write( srOnNativeSeg['super_resolution'], '/tmp/tempI.nii.gz' )
ants.image_write( gtSR['super_resolution_segmentation'], '/tmp/tempGT.nii.gz' )
ants.image_write( srseg['probseg'], '/tmp/tempSRSeg.nii.gz' )
ants.image_write( nativeseg['probseg'], '/tmp/tempORSeg.nii.gz' )
