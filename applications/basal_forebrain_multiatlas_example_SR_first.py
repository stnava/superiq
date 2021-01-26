# this script assumes the image have been brain extracted
import os.path
from os import path
import glob as glob

# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

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
from superiq import list_to_string

# user definitions here
tdir = "/Users/stnava/code/super_resolution_pipelines/data/OASIS30/"
brains = glob.glob(tdir+"Brains/*")
brains.sort()
brains = brains[0:8] # shorten this for the test application
brainsSeg = glob.glob(tdir+"Segmentations/*")
brainsSeg.sort()
brainsSeg = brainsSeg[0:8] # shorten this for this test application
templatefilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template.nii.gz"
templatesegfilename = "/Users/stnava/code/super_resolution_pipelines/template/adni_template_dkt_labels.nii.gz"
sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
output_filename = "outputs3/ADNI_BF"
wlab = [ 75, 76 ] # basal forebrain in OASIS

# input data
imgIn = ants.image_read( infn )
template = ants.image_read( templatefilename )
templateL = ants.image_read( templatesegfilename )
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

havelabels = check_for_labels_in_image( wlab, templateL )

if not havelabels:
    raise_error_here

# expected output data
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg_init = output_filename  +  "_SR_seginit.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"


# first, run registration - then do SR in the local region
if not 'reg' in locals():
    print("SyN begin")
    reg = ants.registration( imgIn, template, 'SyN' )
    forward_transforms = reg['fwdtransforms']
    initlab0 = ants.apply_transforms( imgIn, templateL,
          forward_transforms, interpolator="genericLabel" )
    print("SyN done")


srseg = super_resolution_segmentation_per_label(
    imgIn = imgIn,
    segmentation = initlab0,
    upFactor = [2,2,2],
    sr_model = mdl,
    segmentation_numbers = wlab,
    dilation_amount = 12,
    verbose = True
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
        submask_dilation=20,  # a parameter that should be explored
        searcher= 2,  # double this for SR
        radder=3,  # double this for SR
        reg_iterations=[100,50,10], # fast test
        syn_sampling = 2,
        syn_metric = 'CC',
        max_lab_plus_one=False,
        output_prefix=output_filename,
        verbose=True,
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
# write anything else out here e.g. label geometry etcetera
# output_filename_sr_seg_csv
