# this script assumes the image have been brain extracted
import os.path
from os import path

threads = os.environ['cpu_threads']
# set number of threads - this should be optimized per compute instance
os.environ["TF_NUM_INTEROP_THREADS"] = threads
os.environ["TF_NUM_INTRAOP_THREADS"] = threads
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

import ants
import antspynet
import tensorflow as tf
import os
import sys
import pandas as pd
import numpy as np

from superiq.pipeline_utils import *

def main(input_config):
    c = LoadConfig(input_config)
    tdir = "data"
    image_path = get_s3_object(c.input_bucket, c.input_value, tdir)
    input_image = ants.image_read(image_path)

    if not os.path.exists(c.output_folder):
        os.makedirs(c.output_folder)
    output_filename = c.output_folder + "/"

    ukbb = antspynet.get_antsxnet_data("biobank")
    template = ants.image_read(ukbb)
    btem = antspynet.brain_extraction(template, 't1')
    template = template * btem

    run_extra=True

    def reg_bxt( intemplate, inimg, inbxt, bxt_type, txtype, dilation=0 ):
        inbxtdil = ants.iMath( inbxt, "MD", dilation )
        img = ants.iMath( inimg * inbxt, "TruncateIntensity", 0.0001, 0.999)
        imgn4 = ants.n3_bias_field_correction(img, downsample_factor=4)
        rig = ants.registration(
                intemplate,
                imgn4,
                txtype,
                aff_iterations=(10000, 500, 0, 0),
            )
        if dilation > 0:
            rigi = ants.apply_transforms( intemplate, inimg * inbxtdil, rig['fwdtransforms'] )
        else:
            rigi = ants.apply_transforms( intemplate, inimg, rig['fwdtransforms'] )
        rigi = ants.iMath( rigi, "Normalize")
        rigi = ants.n3_bias_field_correction( rigi, downsample_factor=4 )
        bxt = antspynet.brain_extraction(rigi, bxt_type )
        if bxt_type == 't1combined':
            bxt = ants.threshold_image( bxt, 2, 3 )
        bxt = ants.apply_transforms(
                fixed=inimg,
                moving=bxt,
                transformlist=rig['invtransforms'],
                whichtoinvert=[True,],
            )
        return bxt

    b0 = antspynet.brain_extraction(input_image, 't1')
    rbxt1 = reg_bxt( template, input_image, b0, 't1', 'Rigid', dilation=0 )
    rbxt2 = reg_bxt( template, input_image, rbxt1, 't1', 'Rigid', dilation=0  )
    rbxt3 = reg_bxt( template, input_image, rbxt2, 't1', 'Rigid', dilation=0 )
    rbxt3 = ants.threshold_image( rbxt3, 0.5, 2. ).iMath("GetLargestComponent")
    rbxt4 = reg_bxt( template, input_image, rbxt3, 't1combined', 'Rigid', dilation=0 )
    if run_extra:
        rbxt5 = reg_bxt( template, input_image, rbxt4, 't1combined', 'Rigid', dilation=25 )
        img = ants.iMath(input_image * rbxt5, "TruncateIntensity", 0.0001, 0.999)
        imgn4 = ants.n4_bias_field_correction(img, shrink_factor=4)
        syn=ants.registration(template, imgn4, "SyN" )
        bxt = ants.apply_transforms( imgn4, btem, syn['invtransforms'], interpolator='nearestNeighbor')
        bxt = bxt * rbxt5
    else:
        bxt = rbxt4

    img = ants.iMath(input_image * bxt, "TruncateIntensity", 0.0001, 0.999)
    bxton4 = ants.n4_bias_field_correction(img, shrink_factor=4 )
    plot_path = 'outputs/bxtoplot.png'
    ants.plot(
        bxton4,
        axis=2,
        filename=plot_path
    )
    output_filename = c.output_folder + "/"

    ants.image_write( bxton4, output_filename + 'n4brain.nii.gz')

    bxt_lgm = ants.threshold_image(bxt, 0.5, 1)
    bxtvol = ants.label_geometry_measures( bxt_lgm )
    bxtvol.to_csv( output_filename + 'brainvol.csv' )

    handle_outputs(
        c.input_value,
        c.output_bucket,
        c.output_prefix,
        c.process_name,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
