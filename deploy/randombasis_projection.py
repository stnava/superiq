import os.path
from os import path

import sys
import ants
import numpy as np
import random
import functools
from operator import mul
from sklearn.utils.extmath import randomized_svd
import ia_batch_utils as batch
import pandas as pd
from multiprocessing import Pool

# for repeatability
np.random.seed(42)

def myproduct(lst):
    return( functools.reduce(mul, lst) )


def main(input_config):
    random.seed(0)
    c = input_config
    nvox = c.nvox
    nBasis = c.nbasis
    X = np.random.rand( nBasis*2, myproduct( nvox ) )

    U, Sigma, randbasis = randomized_svd(
        X,
        n_components=nBasis,
        random_state=None
    )
    if randbasis.shape[1] != myproduct(nvox):
        raise ValueError("columns in rand basis do not match the nvox product")

    randbasis = np.transpose( randbasis )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    if hasattr(c, 'template_bucket'):
        templatefn = batch.get_s3_object(c.template_bucket, c.template_key, 'data')
    imgfn = batch.get_s3_object(c.input_bucket, c.input_value, 'data')
    img = ants.image_read(imgfn)
    norm = ants.iMath(img, 'Normalize')
    resamp = ants.resample_image(norm, nvox, use_voxels=True)
    if hasattr(c, 'registration_transform') and hasattr(c, "template_bucket"):
        accepted_transforms = ["Rigid", "Affine", "Similarity", "SyN"]
        if c.registration_transform in accepted_transforms:
            registration_transform = c.registration_transform
        else:
            raise ValueError(f"Expected registration_transform values [{*accepted_transforms,}], not {c.registration_transform}")
        template = ants.image_read( templatefn ).resample_image(nvox, use_voxels=True)
        resamp = ants.registration( template, resamp, registration_transform )['warpedmovout']
    imat = ants.image_list_to_matrix([resamp], resamp*0+1)
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    imgsum = resamp.sum()

    record = {}
    uproj_counter = 0
    for i in uproj[0]:
        uproj_counter += 1
        name = "RandBasisProj" + str(uproj_counter).zfill(2)
        record[name] = i
    uprojpos_counter = 0
    for i in uprojpos[0]:
        uprojpos_counter += 1
        name = "RandBasisProjPos" + str(uprojpos_counter).zfill(2)
        record[name] = i
    df = pd.DataFrame(record, index=[0])
    fields = [i for i in df.columns if i.startswith('RandBasis')]
    batch.record_random_basis_projections(
        c.input_bucket,
        c.input_value,
        df,
        c.resolution,
        c.batch_id,
    )

if __name__ == "__main__":
    config = sys.argv[1]
    c = batch.LoadConfig(config)
    main(c)
