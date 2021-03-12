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
from superiq import list_to_string
# from pipeline_utils import *


# user definitions here
tdir = "/Users/stnava/data/BiogenSuperRes/CIT168_Reinf_Learn_v1/"
sdir = "/Users/stnava/Downloads/temp/adniin/002_S_4473/20140227/T1w/000/brain_ext/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
tfn = tdir + "CIT168_T1w_700um_pad.nii.gz"
tfnl = tdir + "det_atlas_25_pad.nii.gz"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"

# config handling
output_filename = "outputs/EXAMPLE4"
# input data
imgIn = ants.image_read( infn )
imgIn = ants.denoise_image( imgIn, noise_model='Rician' )
imgIn = ants.iMath( imgIn, "TruncateIntensity", 0.00001, 0.995 )
template = ants.image_read(tfn)
templateL = ants.image_read(tfnl)
mdl = tf.keras.models.load_model( model_file_name ) # FIXME - parameterize this

# expected output data
output_filename_jac = output_filename + "_jacobian.nii.gz"
output_filename_seg = output_filename + "_ORseg.nii.gz"
output_filename_sr = output_filename + "_SR.nii.gz"
output_filename_sr_seg = output_filename  +  "_SR_seg.nii.gz"
output_filename_sr_seg_csv = output_filename  + "_SR_seg.csv"
output_filename_warped = output_filename  + "_warped.nii.gz"

is_test=False

if not 'reg' in locals():
    regits = (600,600,600,200,50)
    verber=False
    if is_test:
        regits=(600,600,0,0,0)
        verber=True
    reg = ants.registration( template, imgIn,
#        type_of_transform="TV[2]",        grad_step = 1.4,
        type_of_transform="SyN",        grad_step = 0.20,
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=regits, verbose=verber )
    print("SyN Done")

ants.image_write( reg['warpedmovout'], output_filename_warped )
myjacobian=ants.create_jacobian_determinant_image( template, reg['fwdtransforms'][0], True )
ants.image_write( myjacobian, output_filename_jac )
print( myjacobian.min() )
print(output_filename_jac)
inv_transforms = reg['invtransforms']
initlab0 = ants.apply_transforms( imgIn, templateL, inv_transforms, interpolator="nearestNeighbor" )
g1 = ants.label_geometry_measures(initlab0,imgIn)
sr_params = { 'upFactor':[2,2,2], 'dilation_amount':8, 'verbose':False}
mynums=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
if is_test:
    sr_params = { 'upFactor':[2,2,2], 'dilation_amount':2, 'verbose':True}
    mynums=[1,2]
srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = mynums,
        dilation_amount = sr_params['dilation_amount'],
        verbose = sr_params['verbose']
    )
g2 = ants.label_geometry_measures(srseg['super_resolution_segmentation'],srseg['super_resolution'])
ants.image_write( srseg['super_resolution'], output_filename_sr )
ants.image_write(srseg['super_resolution_segmentation'], output_filename_sr_seg )
