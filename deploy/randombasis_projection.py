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

from multiprocessing import Pool

# for repeatability
np.random.seed(42)

def myproduct(lst):
    return( functools.reduce(mul, lst) )


def main(input_config):
    random.seed(0)
    c = input_config
    print(c.input_value)
    randbasis = batch.get_s3_object(c.rha_bucket, c.rha_key, 'data')
    randbasis = ants.image_read(randbasis).numpy()
    nvox = [c.nvox]*c.dimensionality
    print(f"Random basis dimensions: {randbasis.ndim}")
    print(f"Basis prod(nvox): {myproduct(nvox)}")
    if randbasis.ndim != myproduct(nvox):
        raise ValueError("Rand basis dimensions to match the voxel product")
    nBasis = c.nbasis
    X = np.random.rand( nBasis*2, myproduct( nvox ) )

    U, Sigma, randbasis = randomized_svd(X,
                                  n_components=nBasis,
                                  random_state=None)

    accepted_transforms = ["Rigid", "Affine", "Similarity", "SyN"]
    if c.registration_transform in accepted_transforms:
        registration_transform = c.registration_transform
    else:
        raise ValueError(f"Expected registration_transform values [{*accepted_transforms,}], not {c.registration_transform}")
    randbasis = np.transpose( randbasis )
    rbpos = randbasis.copy()
    rbpos[rbpos<0] = 0
    templatefn = batch.get_s3_object(c.template_bucket, c.template_key, 'data')
    imgfn = batch.get_s3_object(c.input_bucket, c.input_value, 'data')
    img = ants.image_read(imgfn)
    norm = ants.iMath(img, 'Normalize')
    resamp = ants.resample_image(norm, nvox, use_voxels=True)
    #if registration_transform is not None and templatefn is not None: # FIXME - check this
    template = ants.image_read( templatefn ).resample_image(nvox, use_voxels=True)
    resamp = ants.registration( template, resamp, registration_transform )['warpedmovout']
    imat = ants.image_list_to_matrix([resamp], resamp*0+1)
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    imgsum = resamp.sum()

    #outputs
    #split = c.input_value.split('/')
    #subject = split[2]
    #date = split[3]
    #image_id = split[5]
    #record = {
    #    'Subject.ID':subject,
    #    "Date":date,
    #    'Image.ID':image_id,
    #    "RandBasisImgSum": imgsum,
    #}
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
    batch.write_outputs(
        c.input_value,
        c.process_name,
        imgfn,
        df,
        c.input_bucket,
        c.input_value,
        c.resolution,
        c.batch_id,
        fields=fields
    )

# FIXME - this process should be generalized for 2D, 3D (maybe) 4D images
# and with whatever project / modality we have on hand.

if __name__ == "__main__":
    config = sys.argv[1]
    c = batch.LoadConfig(config)
    main(c)
