import ia_batch_utils as batch
import pandas as pd

def get_data(procid):
    df = pd.DataFrame()
    for i in procid:
        new = batch.collect_data(i, '')
        df = pd.concat([df,new])
    df = batch.pivot_data(df)
    return df

def join_all(dfs, how):
    df = dfs[0]
    for d in dfs[1:]:
        merge = pd.merge(df, d, on='originalimage', how=how, suffixes=('', "_y"))
        cols = [i for i in merge.columns if i.endswith('_y')]
        merge.drop(cols, axis=1, inplace=True)
        df = merge

    return df

proc = {
    "bxt":["B143"],
    "rbp":["4350"],
    "hemi_sr":["F210"],
    "bf_star":[],
    "deep_dkt":[],
    "deep_hippo":[],
    "hemi_reg":[],
}

bxt = get_data(proc['bxt'])
bxt['originalimage'] = [i + '.nii.gz' for i in bxt['originalimage']]
rbp = get_data(proc['rbp'])
hemi_sr = get_data(proc['hemi_sr'])


data = [bxt, rbp, hemi_sr]
vols = join_all(data, "left")
meta = pd.read_csv('s3://eisai-basalforebrainsuperres2/metadata/full_metadata_20210208.csv')
meta['originalimage'] = meta['filename']
df = join_all([meta,vols], "right")
df.to_csv('s3://eisai-basalforebrainsuperres2/eisai_20210524_with_metadata.csv')
