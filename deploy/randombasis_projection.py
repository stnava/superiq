import os.path
from os import path

import sys
import numpy as np

from superiq.pipeline_utils import *
from multiprocessing import Pool

def main(input_config):
    try:
        c = input_config
        print(c.input_value)
        nvox = 128 # FIXME - should parameterize this with dimensionality eg (128,128) for 2D
        registration_transform = None # FIXME should allow Rigid, Affine, Similarity, SyN
        randbasis = get_s3_object(c.rha_bucket, c.rha_key, 'data')
        randbasis = ants.image_read(randbasis).numpy()
        rbpos = randbasis
        rbpos[rbpos<0] = 0
        templatefn = get_s3_object(c.input_bucket, c.input_value, 'data') # FIXME
        imgfn = get_s3_object(c.input_bucket, c.input_value, 'data')
        img = ants.image_read(imgfn)
        norm = ants.iMath(img, 'Normalize')
        resamp = ants.resample_image(norm, nvox, use_voxels=True)
        if registration_transform is not None and templatefn is not None: # FIXME - check this
            template = ants.image_read( templatefn )
            resamp = ants.registration( template, resample, registration_transform )['warpedmovout']
        imat = ants.image_list_to_matrix([resamp], resamp*0+1)
        uproj = np.matmul(imat, randbasis)
        uprojpos = np.matmul(imat, rbpos)
        imgsum = resamp.sum()
        #outputs
        split = c.input_value.split('/')
        subject = split[2]
        date = split[3]
        image_id = split[5]
        record = {
            'Subject.ID':subject,
            "Date":date,
            'Image.ID':image_id,
            "RandBasisImgSum": imgsum,
        }
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
        df.to_csv(c.csv_output, index=False)
    except:
        pass

# FIXME - this process should be generalized for 2D, 3D (maybe) 4D images
# and with whatever project / modality we have on hand.

if __name__ == "__main__":
    config = sys.argv[1]
    keys = list_images('mjff-ppmi', 't1_brain_extraction_v2/PPMI/')
    bxt  = [i for i in keys if "nii.gz" in i]
    configs = []
    bucket_prefix = "s3://mjff-ppmi/"
    runs = list_images('mjff-ppmi', 'superres-pipeline-mjff-randbasis/raw/')
    #completed = [bucket_prefix +  i for i in runs]
    completed = []
    for i in bxt:
        c = LoadConfig(config)
        c.input_value = i
        csv_output = f"s3://{c.output_bucket}/{c.output_prefix}raw/{c.input_value.split('/')[5]}.csv"
        c.csv_output = csv_output
        if csv_output not in completed:
            configs.append(c)
        else:
            print(f"skipping: {csv_output}")
    with Pool(30) as p:
        x = p.map(main, configs)
    print("Process Complete")
    bucket_prefix = "s3://mjff-ppmi/"
    results = list_images('mjff-ppmi', 'superres-pipeline-mjff-randbasis/raw/')
    dfs = []
    for i in results:
        df = pd.read_csv(bucket_prefix + i)
        dfs.append(df)
    full = pd.concat(dfs)
    full.to_csv(bucket_prefix + 'superres-pipeline-mjff-randbasis/fullprojs.csv', index=False)
