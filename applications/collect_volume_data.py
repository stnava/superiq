import ia_batch_utils as batch
import pandas as pd

def qa(procid, filename):
    df = pd.DataFrame()
    for i in procid:
        new = batch.collect_data(i, '')
        df = pd.concat([df,new])
    df.to_csv(f's3://invicro-data-outputs/dynamoqa/{filename}-stacked.csv')
    df = batch.pivot_data(df)
    df.to_csv(f's3://invicro-data-outputs/dynamoqa/{filename}-pivoted.csv')
    print("******")
    return df

def join_all(dfs, how):
    df = dfs[0]
    for d in dfs[1:]:
        merge = pd.merge(df, d, on='originalimage', how=how, suffixes=('', "_y"))
        cols = [i for i in merge.columns if i.endswith('_y')]
        merge.drop(cols, axis=1, inplace=True)
        df = merge

    return df

bxt = qa(['07FA'], 'bxt')
bxt['originalimage'] = [i.replace('.nii.gz.nii.gz', '.nii.gz') for i in bxt['originalimage']]
#rdp1 =  qa(['546B'], 'rdp1')
rdp =  qa(['EBF7'], 'rdp')
hemi_sr =  qa(['AA80'], 'hemisr')
meta = pd.read_csv('s3://eisai-basalforebrainsuperres2/metadata/full_metadata_20210208.csv')
meta['originalimage'] = meta['filename']
data = [bxt, rdp, hemi_sr]
vols = join_all(data, "left")
#vols['originalimage'] = [i.replace('.nii.gz.nii.gz', '.nii.gz') for i in vols['originalimage']]
df = join_all([meta,vols], "right")
df.to_csv('s3://eisai-basalforebrainsuperres2/eisai_volume_outputs_with_metadata.csv')
