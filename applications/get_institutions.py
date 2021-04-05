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
      institution_counts.pop('<missing>')
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

def clustering2(df):
      # make a dict for each patno with all the institutions
      clusters = {}
      for i,r in df.iterrows():
            inst = r['institution']
            patno = r['subject_id']
            try:
                  found = clusters[patno]
                  found.append(inst)
                  clusters[patno] = found
            except KeyError:
                  clusters[patno] = [inst,]
      inst_lists = [v for k,v in clusters.items()]
      # cluster institutions
      inst_dicts = {}
      for i in inst_lists:
            try:
                  found = inst_dicts[i[0]]
                  found.append(i)
                  inst_dicts[i[0]] = list(set(found))
            except KeyError:
                  inst_dicts[i[0]] = list(set(i))

      inst = list(df['institution'])
      institution_counts = {k: len(list(v)) for k,v in groupby(inst)}
      institution_counts.pop('<missing>')
      # determine most common name for each group
      old_new_map = {}
      for k,v in inst_dicts.items():
            counts = {}
            for i in v:
                  try:
                        counts[i] = institution_counts[i]
                  except KeyError:
                        continue
            max_key = max(counts, key=counts.get)
            for j in v:
                  old_new_map[j] = max_key
      df['institution_clustered'] = df.apply(lambda x: old_new_map[x['institution']], axis=0)
      return df

      #pairs = []
      #for k,v in clusters.items():
      #      others = {key:val for key,val in cluster.items()}
      #      for k2, v2 in others.items():
      #            if any(items in v for items in v2):
      #                  pairs.append([k,k2])

      #full = [pairs[0],]
      #for i in pairs[1:]:
      #      new_add = []
      #      no_add = []
      #      for f in full:
      #            if i[0] in f or i[1] in m:


if __name__ == "__main__":
      bucket = 'ppmi-image-data'
      prefix = 'NEW_PPMI/DCM_RAW/PPMI/'
      inst = get_institutions(bucket, prefix)
      df = clustering2(inst)
      #df = pd.merge(inst, sub_inst_map, on='subject_id')

      df.to_csv('./institution_map.csv')
