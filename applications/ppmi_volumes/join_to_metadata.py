import pandas as pd
import boto3

name = "-20210401"
metadata = "s3://ppmi-metadata/derived_tables/demog_ppmi_built_07042021.csv"
cst = f's3://mjff-ppmi/volume_measures/direct_reg_seg_ppmi_volumes-mjff{name}-cst.csv'
dir_reg_seg = f's3://mjff-ppmi/volume_measures/direct_reg_seg_ppmi_volumes-mjff{name}.csv'

metadata_df = pd.read_csv(metadata)
#cst_df = pd.read_csv(cst)
drs = pd.read_csv(dir_reg_seg)

drs['Image.ID'] = [int(i.replace('I', '')) for i  in drs['Repeat']]

join = pd.merge(metadata_df, drs, on='Image.ID', how='left')

output = f's3://mjff-ppmi/volume_measures/VolumesDemogPPMI{name}.csv'
join.to_csv(output)
