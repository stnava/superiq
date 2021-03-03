import pandas as pd
import boto3

file_ = "stacked_bf_volumes_dkt.csv"
df = pd.read_csv(file_)

df['concat'] = \
      df['Measure'] + \
      df['Label'].astype(str) + \
      df['Project'] + \
      df['Subject'] + \
      df['Date'].astype(str) + \
      df['Modality'] + \
      df['Repeat'].astype(str) + \
      df['Process'] + \
      df['OriginalOutput'] + \
      df['Resolution']

counts = df['concat'].value_counts()
counts = pd.DataFrame(counts).reset_index()

df = pd.merge(df, counts, how='outer', left_on='concat', right_on='index')
df[df['concat_y']>1].to_csv('more_than_one.csv')
s3 = boto3.client('s3')
s3.upload_file("more_than_one.csv", "eisai-basalforebrainsuperres2", "more_than_one.csv")
