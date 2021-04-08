import os.path
from os import path

import sys
import numpy as np

from superiq.pipeline_utils import *
from multiprocessing import Pool

def main(input_config):
    c = input_config
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
    uproj = np.matmul(imat, randbasis)
    uprojpos = np.matmul(imat, rbpos)
    imgsum = resamp.sum()
    #outputs
    split = c.input_value.split('/')
    #subject = split[2]
    image_id = split[5]
    record = {'Image.ID':image_id,"RandBasisImgSum": imgsum,}
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

if __name__ == "__main__":
    config = sys.argv[1]
    keys = list_images('mjff-ppmi', 't1_brain_extraction_v2/PPMI/')
    bxt  = [i for i in keys if "nii.gz" in i]
    configs = []
    for i in bxt:
        c = LoadConfig(config)
        c.input_value = i
        csv_output = f"s3://{c.output_bucket}/{c.output_prefix}raw/{c.input_value.split('/')[5]}.csv"
        c.csv_output = csv_output
        configs.append(c)
    with Pool(30) as p:
        x = p.map(main, configs)
    bucket_prefix = "s3://mjff-ppmi/"
    results = list_images('mjff-ppmi', 'superres-pipeline-mjff-randbasis/raw/')
    dfs = []
    for i in results:
        print(i)
        df = pd.read_csv(bucket_prefix + i)
        dfs.append(df)
    full = pd.concat(dfs)
    full.to_csv(bucket_prefix + 'superres-pipeline-mjff-randbasis/fullprojs.csv', index=False)

