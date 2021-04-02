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
      return df

from itertools import groupby

def clustering(df):
      subjects = list(set(df['subject_id']))
      inst = list(df['Institution'])
      inst.sort()
      institution_counts = {k: len(list(v)) for k,v in groupby(inst)}
      sub_inst_map = []
      for s in subjects:
            subject_inst = {}
            sublist = df[df['subject_id']==s]
            institutions = list(set(sublist['Institution']))
            institutions.sort()
            sub_institution_counts = {k: len(list(v)) for k,v in groupby(institutions)}
            new_counts = {}
            for i in sub_institution_counts:
                  if i in institution_counts:
                        new_counts[i] = sub_institution_counts[i] + institution_counts[i]
                  else:
                        new_counts[i] = sub_institution_counts[i]
            max_key = max(new_counts, key=new_counts.get)
            sub_inst = [s, max_key]
            sub_inst_map.append(sub_inst)

      df = pd.DataFrame(sub_inst_map, columns = ['subject_id', 'clusteredInstitution'])
      return df
            #group_lists.append(institutions)
      #for i in institutions:
      #      for g in group_lists:
      #            if

if __name__ == "__main__":
      bucket = 'ppmi-image-data'
      prefix = 'NEW_PPMI/DCM_RAW/PPMI/'
      inst = get_institutions(bucket, prefix)
      sub_inst_map = clustering(inst)
      df = pd.merge(inst, sub_inst_map, on='subject_id')

      df.to_csv('./institution_map.csv')
