import ia_batch_utils as batch
import pandas as pd

def get_data(procid, name=""):
    df = pd.DataFrame()
    for i in procid:
        new = batch.collect_data(i, '')
        df = pd.concat([df,new])
    df.to_csv(f"s3://eisai-basalforebrainsuperres2/test_stack_{name}.csv")
    dupe_fields = [
        'key',
        'originalimage',
        'label',
        'process',
        'version',
        'name',
        'resolution',
        'batchid',
    ]
    df.drop_duplicates(dupe_fields, inplace=True)
    df = batch.pivot_data(
        df,
        index_fields=[
            'project',
            'subject',
            'date',
            'modality',
            'repeat',
            'originalimage',
        ],
        column_name_fields=[
            'key',
            'label',
            'resolution',
            'process',
            'version',
            'name',
        ],
        exclude_fields=['hashfields', 'batchid','extension', 'hashid'],
    )
    df.to_csv(f"s3://eisai-basalforebrainsuperres2/test_pivot_{name}.csv")
    return df

def join_all(dfs, how):
    df = dfs[0]
    for d in dfs[1:]:
        merge = pd.merge(df, d, on='originalimage', how=how, suffixes=('', "_y"))
        cols = [i for i in merge.columns if i.endswith('_y')]
        merge.drop(cols, axis=1, inplace=True)
        df = merge

    return df

def fix_localjlf(df):
    df['originalimage'] = [i.replace('-brain_extraction-V0-n4brain',".nii.gz") for i in df['originalimage']]
    return df

proc = {
    "bxt":["B143"],
    "rbp":["D00E"],
    "hemi_sr":["F210"],
    "bf_star_sr":["C1C1"],
    "bf_star_or":["120F"],
    #"deep_dkt":["C4FF", "A2B5"],
    "deep_dkt":["8740", "EFFA"],
    "deep_hippo":['56D6'],
    "localjlf": ['04C0']
}

bxt = get_data(proc['bxt'])
print(f'bxt: {bxt.shape}')
bxt['originalimage'] = [i + '.nii.gz' for i in bxt['originalimage']]
rbp = get_data(proc['rbp'])
print(f'rbp: {rbp.shape}')
hemi_sr = get_data(proc['hemi_sr'])
print(f'hemi_sr: {hemi_sr.shape}')
deep_hippo = get_data(proc['deep_hippo'], name="deep_hippo")
print(f'deep_hippo: {deep_hippo.shape}')
deep_dkt = get_data(proc['deep_dkt'], name="deep_dkt")
print(f'deep_dkt: {deep_dkt.shape}')
bf_star_sr = get_data(proc['bf_star_sr'])
print(f'bf_star_sr: {bf_star_sr.shape}')
bf_star_or = get_data(proc['bf_star_or'])
print(f'bf_star_or: {bf_star_or.shape}')
localjlf = get_data(proc['localjlf'], name="localjlf")
localjlf = fix_localjlf(localjlf)
print(f'localjlf: {localjlf.shape}')



data = [bxt, rbp, hemi_sr, deep_hippo, bf_star_or, deep_dkt, bf_star_sr, localjlf]
vols = join_all(data, "left")
meta = pd.read_csv('s3://eisai-basalforebrainsuperres2/metadata/full_metadata_20210208.csv')
meta['originalimage'] = meta['filename']
df = join_all([meta,vols], "right")
df.to_csv('s3://eisai-basalforebrainsuperres2/eisai_20210524_with_metadata.csv', index=False)
