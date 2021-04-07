# this script assumes the image have been brain extracted
import os.path
from os import path

#threads = os.environ['cpu_threads']
## set number of threads - this should be optimized per compute instance
#os.environ["TF_NUM_INTEROP_THREADS"] = threads
#os.environ["TF_NUM_INTRAOP_THREADS"] = threads
#os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

#import ants
#import antspynet
#import tensorflow as tf
import sys
import numpy as np

from superiq import super_resolution_segmentation_per_label
from superiq import list_to_string
from superiq.pipeline_utils import *

def main(input_config):
    c = LoadConfig(input_config)
    nvox = 128
    randbasis = get_s3_object(c.rha_bucket, c.rha_key, 'data')
    randbasis = ants.image_read(randbasis).numpy()
    rbpos = randbasis
    rbpos[rbpos<0] = 0
    imgfn = get_s3_object(c.input_bucket, c.input_value, 'data')
    img = ants.image_read(imgfn)
    norm = ants.iMath(img, 'Normalize')
    resamp = ants.resample_image(norm, [nvox]*3, use_voxels=True)
    imat = ants.image_list_to_matrix([resamp], resamp*0+1)
    print(f'imat.shape: {imat.shape}')
    uproj = np.matmul(imat, randbasis)
    print(f'uproj.shape: {uproj.shape}')
    uprojpos = np.matmul(imat, rbpos)
    print(f'uproj.shape: {uprojpos}')
    imgsum = resamp.sum()
    print(f'imgsum: {imgsum}')
    #nvox_cubed = nvox * nvox * nvox
    #nelts = nvox_cubed * nv
    #seed = np.random.seed(0)
    #mat = np.random.normal(size=(nv, nvox_cubed))
    #print(mat.shape)
    #randbasis = np.linalg.svd(mat)

    #tdir = "data"
    #ifn = get_s3_object(c.input_bucket, c.input_value, tdir)
    #img = ants.image_read( ifn )



    #if not os.path.exists('outputs'):
    #    os.makedirs('outputs')

    #handle_outputs(
    #    c.input_value,
    #    c.output_bucket,
    #    c.output_prefix,
    #    c.process_name,
    #)

    ## FIXME - write out label geometry measures for:
    ## ants.threshold_image( CSTL, 0.5, 1 )
    ## ants.threshold_image( CSTR, 0.5, 1 ) and the SMA segmentation
    ## just do this at super-resolution
    ## also write out the full DKT image

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
