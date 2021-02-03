import ants
from superiq.pipeline_utils import *
import pandas

bucket = "eisai-basalforebrainsuperres2"
metadata_key = "metadata/full_metadata.csv"
prefix = "s3://eisai-basalforebrainsuperres2/superres-pipeline-20210202/ADNI/"
keys = list_images(bucket, prefix)

keys = [i for i in keys if "basalforebrain" in i]
keys = [i for i in keys if ".csv" in i]

for k in keys:
	
 

for i,r in lgms.iterrows():
        label = int(r['Label'])
        if label_map_params is not None:
            label = label_map_dict[label]['LabelName']
            clean_label = label.replace(' ', '_')
        else:
            clean_label = label
        fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared'] 
        select_data = r[fields] 
        values = select_data.values
        field_values = zip(fields, values)
        
        for f in field_values:
            new_df = {}
            new_df['Measure'] = f[0]
            new_df['Value'] = f[1]
            new_df['Process'] = process 
            new_df['Resolution'] = resolution
            if direction is not None: 
                new_df['Side'] = direction
            else:
                new_df['Side'] = 'full'
            new_df['Label'] = clean_label
            new_df = pd.DataFrame(new_df, index=[0])
            new_rows.append(new_df)