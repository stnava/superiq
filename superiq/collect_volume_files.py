import ants
from superiq.pipeline_utils import *
import pandas
import os
import boto3
import multiprocessing as mp


class VolumeData:

    def __init__(self, bucket, prefix, upload_prefix, cache=False):
        self.bucket = bucket
        self.prefix = prefix
        self.cache = cache
        self.upload_prefix = upload_prefix

    def _filter_keys(self):
        print("====> Getting keys")
        keys = list_images(self.bucket, self.prefix)
        keys = [i for i in keys if i.endswith('.csv')]
        return keys

    def upload_file(self, filename):
        s3 = boto3.client('s3')
        s3.upload_file(
            filename,
            self.bucket,
            self.upload_prefix + filename,
        )

    def look_for_file(self, filename):
        s3 = boto3.client('s3')
        local_path = f"/tmp/{filename}.csv"
        try:
            s3.download_file(self.bucket, self.upload_prefix + filename, local_path)
            return local_path
        except Exception as e:
            print(e)
            print("File not found running new")
            return None

    def _get_files(self, k):
        bucket = self.bucket
        path = get_s3_object(bucket, k, "tmp")
        df = pd.read_csv(path)
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
        filename = path.split('/')[-1]
        split = filename.split('-')
        name_list = ["Project", "Subject", "Date", "Modality", "Repeat", "Process"]
        zip_list = zip(name_list, split)
        for i in zip_list:
            df[i[0]] = i[1]
        df['OriginalOutput'] = "-".join(split[:5]) + ".nii.gz"
        if  "_OR_" in k:
            df['Resolution'] = "OR"
        else:
            df['Resolution'] = "SR"
        os.remove(path)
        return df

    def stack_volumes(self, stack_filename):
        print("====> Stacking volumes")
        if self.cache:
            local_file = self.look_for_file(stack_filename)
            stack_filename =  local_file
        else:
            local_file = None
        if local_file is None:
            with mp.Pool() as p:
                keys = self._filter_keys()
                dfs = p.map(self._get_files, keys)
            stacked = pd.concat(dfs)
            stacked.to_csv(stack_filename, index=False)
            self.upload_file(stack_filename)
        else:
            print("====> Cached stacked_volumes found, skipping")
        return stack_filename

    def pivot_data(self, stack_filename, pivot_filename):
        print("====> Pivoting Data")
        if self.cache:
            local_file = self.look_for_file(pivot_filename)
            pivot_filename =  local_file
        else:
            local_file = None
        if local_file is None:
            df = pd.read_csv(stack_filename)
            #df['Name'] = [i.split('.')[0] for i in df['Name']]
            pivoted = df.pivot(
                index=['Project','Subject','Date', 'Modality', 'Repeat',"OriginalOutput"],
                columns=['Measure', 'Label',"Resolution",'Process'])

            columns = []
            for c in pivoted.columns:
                cols = [str(i) for i in c]
                column_name = '-'.join(cols[1:])
                columns.append(column_name)

            pivoted.columns = columns
            pivoted.reset_index(inplace=True)
            final_csv = pivoted

            final_csv['Repeat'] = [str(i).zfill(3) for i in final_csv['Repeat']]
            final_csv.to_csv(pivoted_filename, index=False)
            self.upload_file(pivot_filename)
        else:
            print("====> Cached pivoted_volumes found, skipping")
        return pivot_filename

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
    bucket = "eisai-basalforebrainsuperres2"
    metadata_key = "volume_measures/data_w_metadata_v01.csv"
    version = "dkt"
    prefix = f"superres-pipeline-{version}/ADNI/"
    stack_filename = f'stacked_bf_volumes_{version}.csv'
    pivoted_filename = f'pivoted_bf_volumes_{version}.csv'
    merge_filename = f"dkt_with_metdata_{version}.csv"
    upload_prefix = "volume_measures/"

    vd = VolumeData(bucket, prefix, upload_prefix, cache=False)

    local_stack = vd.stack_volumes(stack_filename)

    local_pivot = vd.pivot_data(local_stack, pivoted_filename)

    vd.merge_data_with_metadata(local_pivot, metadata_key, merge_filename)
