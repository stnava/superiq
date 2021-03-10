from superiq import VolumeData

if __name__ == "__main__":
      bucket = "mjff-ppmi"
      #metadata_key = "volume_measures/data_w_metadata_v01.csv"
      version = "ljlf-left"
      prefix = f"superres-pipeline-{version}/"
      stack_filename = f'ppmi_stacked_volumes_{version}.csv'
      pivoted_filename = f'ppmi_pivoted_volumes_{version}.csv'
      #merge_filename = f"dkt_with_metdata_{version}.csv"
      upload_prefix = "volume_measures/"
      vd = VolumeData(bucket, prefix, upload_prefix, cache=False)
      local_stack = vd.stack_volumes(stack_filename)
      local_pivot = vd.pivot_data(local_stack, pivoted_filename)

