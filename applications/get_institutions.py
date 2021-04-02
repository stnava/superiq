import boto3
import pydicom
import pandas as pd
import pickle as pkl

def get_institutions(bucket, prefix):
      s3 = boto3.client('s3')
      pickle_file = "./uni_keys.pkl"
      try:
            uni_keys = pkl.load(open(pickle_file, 'rb'))
      except FileNotFoundError:
            print("Not found making list")
            paginator = s3.get_paginator('list_objects')
            pages = paginator.paginate(Bucket=bucket,Prefix=prefix)
            uni_keys = []
            uni_keys_suffix = []
            for page in pages:
                  conts = page['Contents']
                  for c in conts:
                        k = c['Key']
                        if k.endswith('.dcm'):
                              suffix = k.split("_")[-1]
                              if suffix not in uni_keys_suffix:
                                    uni_keys_suffix.append(suffix)
                                    uni_keys.append(k)
            pkl.dump(uni_keys, open(pickle_file, 'wb'))
      print(len(uni_keys))
      pickle_file = "./keys.pkl"
      try:
            keys = pkl.load(open(pickle_file, 'rb'))
      except FileNotFoundError:
            keys = []
            for i in uni_keys:
                  key_dict = {}
                  key_dict['key'] = i
                  key_dict['subject_id'] = i.split('/')[3]
                  key_dict['image_id'] = i.split('_')[-1].replace('.dcm','')
                  filename = '/tmp/outfile.dcm'
                  s3.download_file(bucket, i, filename)
                  ds = pydicom.dcmread(filename)
                  try:
                        inst = ds.InstitutionName
                        if inst == '':
                              inst = '<missing>'
                  except AttributeError:
                        inst = '<missing>'
                  key_dict['institution'] = inst
                  keys.append(key_dict)
            pkl.dump(keys, open(pickle_file, 'wb'))
      df = pd.DataFrame(keys)
      df.to_csv('./institution_map.csv')


if __name__ == "__main__":
      bucket = 'ppmi-image-data'
      prefix = 'NEW_PPMI/DCM_RAW/PPMI/'
      inst = get_institutions(bucket, prefix)
      #print(inst.head())
      #3007/AX_PD__5_1/2011-05-23_08_41_29.0/S989159/PPMI_3007_MR_AX_PD__5_1_br_raw_20201229105902511_7_S989159_I1393669.dcm'
