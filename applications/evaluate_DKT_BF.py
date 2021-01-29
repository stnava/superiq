import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import os.path
from os import path
import glob as glob
# set number of threads - this should be optimized per compute instance

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

# user definitions here
tdir = "/Users/stnava/data/superiq_data_resources/"
brains = glob.glob(tdir+"segmentation_libraries/OASIS30/Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"segmentation_libraries/OASIS30/Segmentations/*")
brainsSeg.sort()
templatefilename = tdir + "template/adni_template.nii.gz"
templatesegfilename = tdir + "template/adni_template_dkt_labels.nii.gz"
seg_params={'submask_dilation': 8, 'reg_iterations': [100, 100, 20],
'searcher': 0, 'radder': 2, 'syn_sampling': 32, 'syn_metric': 'mattes',
'max_lab_plus_one': True, 'verbose': True}

if not 'doSR' in locals():
    doSR = False

if doSR:
    seg_params={'submask_dilation': 16, 'reg_iterations': [100, 100, 100, 20],
       'searcher': 0, 'radder': 3, 'syn_sampling': 32, 'syn_metric': 'mattes',
        'max_lab_plus_one': True, 'verbose': True}

sr_params={"upFactor": [2,2,2], "dilation_amount": seg_params["submask_dilation"], "verbose":True}
mdl = tf.keras.models.load_model("models/SEGSR_32_ANINN222_3.h5")

# store output data
brainName = []
dicevalLR = []
dicevalHR = []

for k in range( len(overlaps), len( brains ) ):
    print( str(k) + " " + str(doSR ))
    brainsLocal=brains.copy()
    brainsSegLocal=brainsSeg.copy()
    del brainsLocal[k:(k+1)]
    del brainsSegLocal[k:(k+1)]
    wlab = [75,76]
    if doSR:
        locseg = ants.image_read(brainsSeg[k])
        srseg = super_resolution_segmentation_per_label(
            imgIn = ants.image_read(brains[k]),
            segmentation = ants.mask_image( locseg, locseg, level = wlab, binarize=False ),
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = wlab,
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose']
            )
        use_image = srseg['super_resolution']
    else:
        use_image=ants.iMath( ants.image_read(brains[k]), "Normalize" )
    localbf = basalforebrain_segmentation(
        target_image=use_image,
        segmentation_numbers = wlab,
        template = ants.image_read(templatefilename),
        template_segmentation = ants.image_read(templatesegfilename),
        library_intensity=images_to_list(brainsLocal),
        library_segmentation=images_to_list(brainsSegLocal),
        seg_params = seg_params
        )
    mypt = 0.5
    bfsegljlf = ants.threshold_image( localbf['probsum'], mypt, 2.0 )
    if not doSR: # here we do SR on the BF output from LR segmentation vs SR first
        gtseg = ants.image_read( brainsSeg[k] )
        # both GT and Appoximation should undergo same operations
        srseg = super_resolution_segmentation_per_label(
            imgIn = use_image,
            segmentation = bfsegljlf,
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = [1],
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose'] )
        srsegGT = super_resolution_segmentation_per_label(
            imgIn = use_image,
            segmentation = ants.mask_image( gtseg, gtseg, level = wlab, binarize=True ),
            upFactor = sr_params['upFactor'],
            sr_model = mdl,
            segmentation_numbers = [1],
            dilation_amount = sr_params['dilation_amount'],
            verbose = sr_params['verbose'] )
    else:
        gtseg = srseg['super_resolution_segmentation']
    # native resolution
    gtlabel = ants.mask_image( gtseg, gtseg, level = wlab, binarize=True )
    myol = ants.label_overlap_measures( gtlabel, bfsegljlf )
    # super resolution
    gtlabelUp = srsegGT['super_resolution_segmentation']
    myolUp = ants.label_overlap_measures( gtlabelUp, srseg['super_resolution_segmentation'] )
    brainName.append( os.path.splitext( os.path.splitext( os.path.basename( brains[k]) )[0])[0])
    dicevalLR.append(myol["MeanOverlap"][0])
    dicevalHR.append( myolUp["MeanOverlap"][0])
    print( brainName[k] + ": LR: " + str(dicevalLR[k]) + " HR: " +  str(dicevalHR[k]) )
#    ants.image_write( srseg['super_resolution'], '/tmp/tempI.nii.gz' )
#    ants.image_write( gtlabelUp, '/tmp/tempGT.nii.gz' )
#    ants.image_write( srseg['super_resolution_segmentation'], '/tmp/tempSRSeg.nii.gz' )


################################################################################
dict = {'name': brainName, 'DiceLR': diceval, 'DiceHR': dicevalHR}
df = pd.DataFrame(dict)
if not doSR:
  	df.to_csv('/tmp/bf_ol_or.csv')
else:
	df.to_csv('/tmp/bf_ol_sr.csv')
