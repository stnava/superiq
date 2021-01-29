import os
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

import os.path
from os import path
import glob as glob
# set number of threads - this should be optimized per compute instance


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
from superiq import sort_library_by_similarity
from superiq import basalforebrain_segmentation

def images_to_list( x ):
    outlist = []
    for k in range(len(x)):
        outlist.append( ants.image_read( x[k] ) )
    return outlist

# user definitions here
tdir = "/Users/stnava/code/super_resolution_pipelines/data/OASIS30/"
brains = glob.glob(tdir+"Brains/*")
brains.sort()
brainsSeg = glob.glob(tdir+"Segmentations/*")
brainsSeg.sort()
templatefilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template.nii.gz"
templatesegfilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template_dkt_labels.nii.gz"

# to hold the evaluation outputs
overlaps = []

seg_params={'submask_dilation': 10, 'reg_iterations': [100, 50, 0],
'searcher': 0, 'radder': 2, 'syn_sampling': 32, 'syn_metric': 'mattes',
'max_lab_plus_one': True, 'verbose': True}

sr_params={"upFactor": [2,2,2], "dilation_amount": 12, "verbose":True}

if not 'doSR' in locals():
    doSR = False

if doSR:
    mdl = tf.keras.models.load_model("models/SEGSR_32_ANINN222_3.h5")

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
        use_image=ants.image_read(brains[k])
    localbf = basalforebrain_segmentation(
        target_image=use_image,
        segmentation_numbers = wlab,
        template = ants.image_read(templatefilename),
        template_segmentation = ants.image_read(templatesegfilename),
        library_intensity=images_to_list(brainsLocal),
        library_segmentation=images_to_list(brainsSegLocal),
        seg_params = seg_params
        )
    if not doSR:
        gtseg = ants.image_read( brainsSeg[k] )
    else:
        gtseg = srseg['super_resolution_segmentation']
    gtlabel = ants.mask_image( gtseg, gtseg, level = wlab, binarize=True )
    bfseg = ants.threshold_image( localbf['probsum'], 0.30, 2.0 )
    myol = ants.label_overlap_measures(gtlabel,bfseg)
    print(myol)
    overlaps.append( myol )

# FIXME - organize the overlap output and write to csv
