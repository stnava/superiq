import ants
from pipeline_utils import *
import glob
import sys
import pandas as pd
import antspynet
import tensorflow as tf
import numpy as np

# should be:
# 1. segmentation images
# 2. segmentation pngs
# 3. segmentation csvs
# 4 probability images
# 5.  super resolution images

def deep_hippo(
    target_image,
):
    pass

# TG:Import the config file
params = handle_config(sys.argv[1])

# Template internal to antspynet
template = antspynet.get_antsxnet_data( "biobank" )
template = ants.image_read( template )
template = template * antspynet.brain_extraction( template )

# TG:Get from s3
initial_image = get_input_image(
        params['parameters']['input_bucket'],
        params['parameters']['input_value'],
)
input_image_path = get_pipeline_data(
        "bxtreg_n3.nii.gz",
        params['parameters']['output_bucket'],
        params['parameters']['output_prefix'],
        initial_image,
)

# TG: Path to image downloaded with get_pipeline_data
img = ants.image_read( "data/bxtreg_n3.nii.gz" )

avgleft = img * 0
avgright = img * 0
nLoop = 10
for k in range(nLoop):
    rig = ants.registration( template, img, "Rigid" )
    rigi = rig['warpedmovout']
    hipp = antspynet.hippmapp3r_segmentation( rigi, do_preprocessing=False )
    hippr = ants.apply_transforms(
        img,
        hipp,
        rig['fwdtransforms'],
        whichtoinvert=[True],
        interpolator='genericLabel'
    )
    #hipporgeom = ants.label_geometry_measures( hippr )
    avgleft = avgleft + ants.threshold_image( hippr, 2, 2 ) / nLoop
    avgright = avgright + ants.threshold_image( hippr, 1, 1 ) / nLoop


avgright = ants.iMath(avgright,"Normalize")  # output: probability image right
avgleft = ants.iMath(avgleft,"Normalize")    # output: probability image left

# OR Prob Images
leftORprob_path = f"{output_path}hippProbORleft.nii.gz"
rightORprob_path = f"{output_path}/hippProbORright.nii.gz"
ants.image_write(avgleft, leftORprob_path)
ants.image_write(avgright, rightORprob_path)

hippright = ants.threshold_image( avgright, 0.5, 10 ).iMath("GetLargestComponent")
hippleft = ants.threshold_image( avgleft, 0.5, 10.0 ).iMath("GetLargestComponent")

# OR Seg Images
leftORseg_path = f"{output_path}hippSegORleft.nii.gz"
rightORseg_path = f"{output_path}hippSegORright.nii.gz"
ants.image_write(hippleft, leftORseg_path) # output: segmentation image
ants.image_write(hippright, rightORseg_path) # output: segmentation image

# output : make segmentation png for left and right on cropped OR image
# Calls ants.plot_ortho with hippr as the img, and hippleft as the overlay image.
# Croping done in background
#plot_output(avgright, "outputs/hippORleft-ortho.png", hippleft) # output: segmentation image
ants.plot_ortho(
        ants.crop_image(img,hippleft),
        ants.crop_image(hippleft,hippleft),
        filename=f"{output_path}hippORleft-ortho.png",
)
#plot_output(avgleft, "outputs/hippORright-ortho.png", hippright) # output: segmentation image

ants.plot_ortho(
        ants.crop_image(img,hippright),
        ants.crop_image(hippright,hippright),
        filename=f"{output_path}hippORright-ortho.png",
)

get_label_geo( # output: label geo for segmentation image
        labeled_image=hippleft,
        initial_image=ants.image_read(initial_image),
        output_bucket=params['parameters']['output_bucket'],
        output_prefix=params['parameters']['output_prefix'],
        process=params['parameters']['process_name'],
        input_key=params['parameters']['input_value'],
        resolution="OR",
        direction="left",
)

get_label_geo( # output: label geo for segmentation image
        labeled_image=hippright,
        initial_image=ants.image_read(initial_image),
        output_bucket=params['parameters']['output_bucket'],
        output_prefix=params['parameters']['output_prefix'],
        process=params['parameters']['process_name'],
        input_key=params['parameters']['input_value'],
        resolution="OR",
        direction="right",
)

def deep_hippo_SR(inputs):
    dosr = True
    imglist = []
    seglist = []
    if dosr:
        for side in [1,2]:
            if side == 1:
                hippimg = avgright
                sideLabel = "right"
            elif side == 2:
                hippimg = avgleft
                sideLabel = "left"

            model_path = get_model(
                    params['parameters']['model_bucket'],
                    params['parameters']['model_key'],
            )
            mdl = tf.keras.models.load_model( model_path )


            img = ants.iMath( img, "Normalize" ) * 255 - 127.5 # for SR
            hipp1 = ants.threshold_image( hippr, side, side )
            hipp1m = ants.iMath(hipp1,'MD',10)
            imgc = ants.crop_image(img,hipp1m)
            imgch = ants.crop_image(hippimg,hipp1m)
            myarr = np.stack( [imgc.numpy(),imgch.numpy()* 255 - 127.5],axis=3 )
            newshape = np.concatenate( [ [1],np.asarray( myarr.shape )] )
            myarr = myarr.reshape( newshape )
            pred = mdl.predict( myarr )
            imgsr = ants.from_numpy( tf.squeeze( pred[0] ).numpy())
            imgsr = ants.copy_image_info( imgc, imgsr )
            newspc = ( np.asarray( ants.get_spacing( imgsr ) ) * 0.5 ).tolist()
            ants.set_spacing( imgsr,  newspc )
            imgsrh = ants.from_numpy( tf.squeeze( pred[1] ).numpy())
            imgsrh = ants.copy_image_info( imgc, imgsrh )
            ants.set_spacing( imgsrh,  newspc )
            imglist.append( imgsr )
            seglist.append( imgsrh )
            imgsrhb = ants.threshold_image( imgsrh, 0.5, 1.0 ).iMath("GetLargestComponent")


rightSR = ants.iMath(imglist[0], "Normalize")  # output: sr image right
leftSR = ants.iMath(imglist[1], "Normalize")  # output: sr image left
ants.image_write(rightSR, "outputs/hippImgSRright.nii.gz")
ants.image_write(leftSR, "outputs/hippImgSRleft.nii.gz")


avgrightSR = seglist[0]  # output: probability image right
avgleftSR = seglist[1]     # output: probability image left
ants.image_write(avgrightSR, "outputs/hippProbSRright.nii.gz")
ants.image_write(avgleftSR, "outputs/hippProbSRleft.nii.gz")


hipprSRright = ants.threshold_image( avgrightSR, 0.5, 10 ).iMath("GetLargestComponent")
hipprSRleft = ants.threshold_image( avgleftSR, 0.5, 10.0 ).iMath("GetLargestComponent")
ants.image_write(hipprSRright, "outputs/hippSegSRright.nii.gz") # output: segmentation image
ants.image_write(hipprSRleft, "outputs/hippSegSRleft.nii.gz") # output: segmentation image


# output : make segmentation png for left and right on cropped SR image
plot_output(rightSR, "outputs/hippSRright-ortho.png", hipprSRright)
plot_output(leftSR, "outputs/hippSRleft-ortho.png", hipprSRleft)

# output : csv for SR left and right
get_label_geo( # output: label geo for segmentation image
        labeled_image=hipprSRright,
        initial_image=rightSR,
        output_bucket=params['parameters']['output_bucket'],
        output_prefix=params['parameters']['output_prefix'],
        process=params['parameters']['process_name'],
        input_key=params['parameters']['input_value'],
        resolution="SR",
        direction="right",
)

get_label_geo( # output: label geo for segmentation image
        labeled_image=hipprSRleft,
        initial_image=leftSR,
        output_bucket=params['parameters']['output_bucket'],
        output_prefix=params['parameters']['output_prefix'],
        process=params['parameters']['process_name'],
        input_key=params['parameters']['input_value'],
        resolution="SR",
        direction="left",
)



handle_all_outputs(
        initial_image,
        params['parameters']['output_bucket'],
        params['parameters']['output_prefix'],
        params['parameters']['process_name'],
)

