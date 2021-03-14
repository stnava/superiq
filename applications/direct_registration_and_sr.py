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
sdir = "/tmp/PPMI/3068/20110620/T1w/S117935/localjlf/"
model_file_name = "/Users/stnava/code/super_resolution_pipelines/models/SEGSR_32_ANINN222_3.h5"
tfn = tdir + "CIT168_T1w_700um_pad.nii.gz"
tfnl = tdir + "det_atlas_25_pad.nii.gz"
infn = sdir + "ADNI-002_S_4473-20140227-T1w-000-brain_ext-bxtreg_n3.nii.gz"
infn = sdir + "PPMI-3068-20110620-T1w-S117935-localjlf-ppmi_OR.nii.gz"
# config handling
output_filename = "outputs/TEST_"
# input data
imgIn = ants.image_read( infn )
imgIn = ants.denoise_image( imgIn, noise_model='Rician' )
imgIn = ants.iMath( imgIn, "TruncateIntensity", 0.00001, 0.9995 ).iMath("Normalize")
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

is_test=True

if not 'reg' in locals():
    regits = (600,600,600,200,50)
    verber=False
    if is_test:
        regits=(600,600,20,0,0)
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
inv_transforms = reg['invtransforms']
initlab0 = ants.apply_transforms( imgIn, templateL, inv_transforms, interpolator="nearestNeighbor" )
ants.image_write( initlab0, output_filename_seg )
g1 = ants.label_geometry_measures(initlab0,imgIn)
sr_params = { 'upFactor':[2,2,2], 'dilation_amount':8, 'verbose':False}
mynums=list( range(1,17) )

if is_test:
    sr_params = { 'upFactor':[2,2,2], 'dilation_amount':2, 'verbose':True}
    mynums=[7,8,9,10,11]

if not 'srseg' in locals():
    srseg = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab0,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = mynums,
        dilation_amount = sr_params['dilation_amount'],
        max_lab_plus_one = True,
        verbose = sr_params['verbose']
    )

ants.image_write( srseg['super_resolution'], output_filename_sr )

ants.image_write( srseg['super_resolution_segmentation'], output_filename_sr_seg )

derka

if not 'reg2' in locals():
    reg2 = ants.registration( srseg['super_resolution'], template,
        type_of_transform="SyN",        grad_step = 0.20,
        syn_metric='CC',
        syn_sampling=2,
        reg_iterations=regits, verbose=verber )
    print("SyN2 Done")


initlab1 = ants.apply_transforms( imgIn, templateL, reg2['fwdtransforms'],
  interpolator="nearestNeighbor" )

# background = ants.threshold_image( initlab1, 1, max(mynums)-1 )
# bkgdilate = 2
# background = ants.iMath(background,"MD",bkgdilate) - background
# initlab0p1 = initlab0 + background * max(mynums)


if not 'srseg2' in locals():
    srseg2 = super_resolution_segmentation_per_label(
        imgIn = imgIn,
        segmentation = initlab1,
        upFactor = sr_params['upFactor'],
        sr_model = mdl,
        segmentation_numbers = mynums,
        dilation_amount = sr_params['dilation_amount'],
        max_lab_plus_one = True,
        verbose = sr_params['verbose']
    )

derk
derka

probimgs=[]
for k in range(len(srseg['probability_images'])):
    tar = srseg['super_resolution_segmentation']
    temp = ants.resample_image_to_target(srseg['probability_images'][k],tar)
    probimgs.append( temp )

tarmask = ants.threshold_image( initlab0, 1, initlab0.max() ).iMath("MD",bkgdilate)
tarmask = ants.resample_image_to_target( tarmask, tar, interp_type='nearestNeighbor' )
segmat = ants.images_to_matrix(probimgs, tarmask)
finalsegvec = segmat.argmax(axis=0)
finalsegvec2 = finalsegvec.copy()

# mapfinalsegvec to original labels
for i in range(len(probimgs)):
    segnum = mynums[i]
    finalsegvec2[finalsegvec == i] = segnum

outimg = ants.make_image(tarmask, finalsegvec2)
ants.image_write( outimg, output_filename_sr_seg )

derka

# next decide what is "background" based on the sum of the first k labels vs the prob of the last one
firstK = probimgs[0] * 0

for i in range(len(probimgs)):
    firstK = firstK + probimgs[i]

background_prob = ants.resample_image_to_target( background,  tar,interp_type='linear')

segmat = ants.images_to_matrix([background_prob, firstK], tarmask)
bkgsegvec = segmat.argmax(axis=0)
outimg = outimg * ants.make_image(tarmask, bkgsegvec)

ants.image_write( outimg, output_filename_sr_seg )
