import os
threads = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import antspyt1w
import superiq
import pandas as pd
import tensorflow as tf

# example image
img = ants.image_read( "PPMI-41664-20191028-MRI_T1-I1350813-antspyt1w-V0-brain_n4_dnz.nii.gz" )
# this is the default template for antspyt1w
template = ants.image_read( antspyt1w.get_data( "T_template0", target_extension="nii.gz") )
################################################################################
# DESCRIBE how to derive SR from antspyt1w.hierarchical outputs
################################################################################
# basically, two ways: label then SR or SR then label
# here, we define the approach for doing labeling then SR.
# first - map to a different template via hemi_reg (this example: CIT168)
templatebxt = template * antspynet.brain_extraction( template )
templateLR = ants.image_read( antspyt1w.get_data( "T_template0_LR", target_extension="nii.gz") )
cit168lab = ants.image_read( antspyt1w.get_data( "det_atlas_25_pad_LR_adni", target_extension="nii.gz") )
cit168 = ants.image_read( antspyt1w.get_data( "CIT168_T1w_700um_pad_adni", target_extension="nii.gz") )
cit168desc = pd.read_csv( antspyt1w.get_data( "CIT168_Reinf_Learn_v1_label_descriptions_pad", target_extension="csv") )
cithemi = antspyt1w.label_hemispheres( cit168, templatebxt, templateLR, reg_iterations=[200, 50, 2, 0])
# new registration
imgseg = ants.image_read( "PPMI-41664-20191028-MRI_T1-I1350813-antspyt1w-V0-dkt_parc_tissue_segmentation.nii.gz" )
imghemi = ants.image_read( "PPMI-41664-20191028-MRI_T1-I1350813-antspyt1w-V0-dkt_parc_hemisphere_labels.nii.gz")
istest = False
citreg = antspyt1w.hemi_reg( img, imgseg, imghemi, cit168, cithemi,
    output_prefix="PPMI-41664-20191028-MRI_T1-I1350813-CITREG", padding=10,
    labels_to_register=[2, 3, 4, 5], is_test=istest )
# apply transformations to this new template's label set (CIT168 SNc etc)
cit2ppmiL = ants.apply_transforms( img, cit168lab, citreg['synL']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( imghemi, 1, 1  )
cit2ppmiR = ants.apply_transforms( img, cit168lab, citreg['synR']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( imghemi, 2, 2  )
citdfL = antspyt1w.map_segmentation_to_dataframe( "CIT168_Reinf_Learn_v1_label_descriptions_pad", cit2ppmiL )
citdfR = antspyt1w.map_segmentation_to_dataframe( "CIT168_Reinf_Learn_v1_label_descriptions_pad", cit2ppmiR )
# now do SR+seg
mysupmdl = tf.keras.models.load_model( "SEGSR_32_ANINN222_3.h5" )
sup = superiq.super_resolution_segmentation_per_label( img, cit2ppmiL,
    upFactor=[2,2,2],
    sr_model=mysupmdl,
    segmentation_numbers=[7,8,9,10], # should do all on left side
    dilation_amount=6,
    probability_images=None,
    probability_labels=None, max_lab_plus_one=True, verbose=True)
ants.image_write( sup['super_resolution'], '/tmp/tempi.nii.gz' )
ants.image_write( sup['super_resolution_segmentation'], '/tmp/temps.nii.gz' )
# output style should be the same as usual : map to dataframe / pivot, etc.
################################################################################
################################################################################
# next, we define the approach for SR then labeling
################################################################################
# do super resolution on whole brain
brainmask = ants.threshold_image( ants.iMath(img,"Normalize"), 1.e-9, 1.0 )
hemisr = superiq.super_resolution_segmentation_per_label( img, imghemi * brainmask,
    upFactor=[2,2,2],
    sr_model=mysupmdl,
    segmentation_numbers=[1,2] )
imgsr = hemisr[ 'super_resolution' ]
imghemisr = hemisr['super_resolution_segmentation']
imgsegsr = ants.resample_image_to_target( imgseg, imgsr, interp_type='nearestNeighbor')
# do high-res registration and then proceed just as above ...
citregsr = antspyt1w.hemi_reg( imgsr, imgsegsr, imghemisr,
    cit168, cithemi,
    output_prefix="PPMI-41664-20191028-MRI_T1-I1350813-CITREGSR",
    padding=10,
    labels_to_register=[2, 3, 4, 5], is_test=istest )
# apply transformations to this new template's label set (CIT168 SNc etc)
cit2ppmiLSR = ants.apply_transforms( imgsr, cit168lab, citregsr['synL']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( imghemisr, 1, 1  )
cit2ppmiRSR = ants.apply_transforms( imgsr, cit168lab, citregsr['synR']['invtransforms'],
          interpolator='genericLabel' ) * ants.threshold_image( imghemisr, 2, 2  )
citdfLSR = antspyt1w.map_segmentation_to_dataframe( "CIT168_Reinf_Learn_v1_label_descriptions_pad", cit2ppmiLSR )
citdfRSR = antspyt1w.map_segmentation_to_dataframe( "CIT168_Reinf_Learn_v1_label_descriptions_pad", cit2ppmiRSR )


