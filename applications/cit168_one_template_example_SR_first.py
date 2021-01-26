# this script assumes the image have been brain extracted
import os.path
from os import path

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
from superiq import ljlf_parcellation_one_template
from superiq import listToString

# user definitions here
tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
tfn = tdir + "CIT168_T1w_700um.nii.gz"
tfnl = tdir + "det_atlas_25.nii.gz"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
output_filename = "/tmp/outputs3/CITI68"
wlab = range(7,11) # substantia nigra in CIT168
# wlab = [1,2,5,6] # this is putamen, caudate - need multi-atlas
wlab = [2] # substantia nigra in CIT168
# input data
imgIn = ants.image_read( infn )
template = ants.image_read(tfn)
templateL = ants.image_read(tfnl)
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

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

locseg = ljlf_parcellation_one_template(
        srseg['super_resolution'],
        segmentation_numbers=wlab,
        forward_transforms=forward_transforms,
        template=template,
        templateLabels=templateL,
        templateRepeats = 8,
        submask_dilation=6,  # a parameter that should be explored
        searcher= 2,  # double this for SR
        radder=3,  # double this for SR
        reg_iterations=[100,100,50,10], # fast test
        syn_sampling = 2,
        syn_metric = 'CC',
        max_lab_plus_one=True,
        output_prefix=output_filename,
        verbose=True,
    )

ants.image_write( locseg['segmentation'], output_filename_sr_seg )
# write anything else out here e.g. label geometry etcetera
# output_filename_sr_seg_csv
