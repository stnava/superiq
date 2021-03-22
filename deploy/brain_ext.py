# this script assumes the image have been brain extracted
import os.path
from os import path

threads = "8"
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

    img = ants.iMath(input_image, "TruncateIntensity", 0.0001, 0.999)
    imgn4 = ants.n4_bias_field_correction(img, img*0+1, 4)
    rig = ants.registration(
        template,
        imgn4,
        "Affine",
        aff_iterrations=(10000, 500, 50, 0),
    )
    rigi = ants.iMath(rig['warpedmovout'], "Normalize")
    bxt = antspynet.brain_extraction(rigi, 't1combined')
    bxt = ants.threshold_image(bxt, 2, 3)
    bxto = ants.apply_transforms(
        fixed=imgn4,
        moving=bxt,
        transformlist=rig['invtransforms'],
        whichtoinvert=[True,],
    )
    bxton4 = ants.n4_bias_field_correction(img, bxto, 4)
    plot_path = 'outputs/bxtoplot.png'
    ants.plot(
        bxton4 * bxto,
        axis=2,
        filename=plot_path
    )
    output_filename = c.output_folder + "/"

    ants.image_write(bxto, output_filename + 'n4brain.nii.gz')

    handle_outputs(
        c.input_value,
        c.output_bucket,
        c.output_prefix,
        c.process_name,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
