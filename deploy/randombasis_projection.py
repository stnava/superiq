import os.path
from os import path

import sys
import numpy as np

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

if __name__ == "__main__":
    config = sys.argv[1]
    main(config)
