import ants
from superiq.pipeline_utils import *
import pandas
import os
import boto3
import multiprocessing as mp


class VolumeData:

    def __init__(self, bucket, prefix, filter_suffixes, upload_prefix):
        self.bucket = bucket
        self.prefix = prefix
        self.filter_suffixes = filter_suffixes
        self.upload_prefix = upload_prefix

    def stack_volumes(self, stack_filename):
        print("====> Stacking volumes")
        with mp.Pool() as p:
            keys = self._filter_keys(self.filter_suffixes)
            dfs = p.map(self._get_files, keys)
        stacked = pd.concat(dfs)
        stack_filename_key = f"s3://{self.bucket}/{key}"
        stacked.to_csv(stack_filename_key, index=False)
        #key = self.upload_file(stack_filename)
        return stack_filename_key

    def _filter_keys(self, filter_suffix):
        print("====> Getting keys")
        keys = list_images(self.bucket, self.prefix)
        filtered_keys = []
        for fil in filter_suffix:
            keys = [i for i in keys if i.endswith(fil)]
            for k in keys:
                if k not in filtered_keys:
                    filtered_keys.append(k)
        print(len(filtered_keys))
        return filtered_keys

    def _get_files(self, k):
        #path = get_s3_object(bucket, k, "tmp/")
        df = pd.read_csv(f"s3://{self.bucket}/{k}")
        fields = ["Label", 'VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
        df = df[fields]
        new_rows = []
        for i,r in df.iterrows():
            label = int(r['Label'])
            fields = ['VolumeInMillimeters', 'SurfaceAreaInMillimetersSquared']
            select_data = r[fields]
            values = select_data.values
            field_values = zip(fields, values)
            for f in field_values:
                new_df = {}
                new_df['Measure'] = f[0]
                new_df['Value'] = f[1]
                new_df['Label'] = label
                new_df = pd.DataFrame(new_df, index=[0])
                new_rows.append(new_df)

        df = pd.concat(new_rows)
        filename = k.split('/')[-1]
        split = filename.split('-')
        name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process", "Name"]
        zip_list = zip(name_list, split)
        for i in zip_list:
            df[i[0]] = i[1]
        df['OriginalOutput'] = "-".join(split[:5]) + ".nii.gz"
        if "SR" in k:
            df['Resolution'] = "SR"
        else:
            df['Resolution'] = "OR"
        return df

    def pivot_data(self, stack_filename_key, pivot_filename):
        print("====> Pivoting Data")
        df = pd.read_csv(stack_filename_key)
        #df['Name'] = [i.split('.')[0] for i in df['Name']]
        pivoted = df.pivot(
            index=['Project','Subject','Date', 'Modality', 'Repeat',"OriginalOutput"],
            columns=['Measure', 'Label',"Resolution",'Process', "Name"])

        columns = []
        for c in pivoted.columns:
            cols = [str(i) for i in c]
            column_name = '-'.join(cols[1:])
            columns.append(column_name)

        pivoted.columns = columns
        pivoted.reset_index(inplace=True)
        final_csv = pivoted

        pivot_filename_key = f"s3://{self.bucket}/{pivot_filename}"
        final_csv['Repeat'] = [str(i).zfill(3) for i in final_csv['Repeat']]
        final_csv.to_csv(pivot_filename_key, index=False)
        #self.upload_file(pivot_filename)
        return pivot_filename_key

    #def upload_file(self, filename):
    #    s3 = boto3.client('s3')
    #    key = self.upload_prefix + filename
    #    s3.upload_file(
    #        filename,
    #        self.bucket,
    #        key
    #    )
    #    return key



    def merge_data_with_metadata(self,
                                 pivoted_filename,
                                 metadata_key,
                                 merge_filename,
                                 on=['filename','OriginalOutput'],
    ):
        data = pd.read_csv(pivoted_filename)
        meta = get_s3_object(self.bucket, metadata_key, "tmp")
        metadf = pd.read_csv(meta)
        os.remove(meta)
        merge = pd.merge(
            metadf,
            data,
            how="outer",
            left_on=on[0],
            right_on=on[1],
            suffixes=("","_x")
        )
        duplicate_columns = [i for i in merge.columns if i.endswith('_x')]
        merge.drop(duplicate_columns, inplace=True, axis=1)
        merge.to_csv(merge_filename, index=False)
        self.upload_file(merge_filename)


if __name__ == "__main__":
    bucket = "mjff-ppmi"
    #metadata_key = "volume_measures/data_w_metadata_v01.csv"
    version = "bxt"
    prefix = "t1_brain_extraction_v2/"
    upload_prefix = "volume_measures/"
    stack_filename = upload_prefix + f'stacked_volumes_{version}.csv'
    pivoted_filename = upload_prefix + f'pivoted_volumes_{version}.csv'
    #merge_filename = f"dkt_with_metdata_{version}.csv"

    filter_suffixes = ['brainvol.csv']
    vd = VolumeData(bucket, prefix,filter_suffixes, upload_prefix)

    local_stack = vd.stack_volumes(stack_filename)

    local_pivot = vd.pivot_data(local_stack, pivoted_filename)

    #vd.merge_data_with_metadata(local_pivot, metadata_key, merge_filename)
