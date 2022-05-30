from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('US_Accidents_Dec21_updated.csv')

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Hour']=df['Start_Time'].dt.hour

features = ['Severity','Distance(mi)','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Crossing','Junction','Traffic_Signal','Hour']
df_feat=df[features].copy()
df_feat['Crossing'] = df_feat['Crossing'].astype(float)
df_feat['Junction'] = df_feat['Junction'].astype(float)
df_feat['Traffic_Signal'] = df_feat['Traffic_Signal'].astype(float)
df_feat.dropna(inplace=True)
df_feat.drop_duplicates(inplace=True)

df_s1_index = df_feat[df_feat['Severity'] == 1].index
df_s1_values=len(df_s1_index)
df_s1_values
df_s1 = df_feat[df_feat['Severity'] == 1]
df_s2 = df_feat[df_feat['Severity'] == 2]
df_s3 = df_feat[df_feat['Severity'] == 3]
df_s4 = df_feat[df_feat['Severity'] == 4]
df_s2u = df_s2.sample(df_s1_values)
df_s3u = df_s3.sample(df_s1_values)
df_s4u = df_s4.sample(df_s1_values)
df_u = pd.concat([df_s1, df_s2u,df_s3u,df_s4u], axis=0)

target = 'Severity'
y=df_u[target]
x=df_u.drop(target,axis=1)
x=np.asarray(x)
y=np.asarray(y)

pca = PCA( n_components=2)

pca.fit(x)

x_hat = pca.transform (x)

plt.figure(figsize=(10, 10))

for k in range(1,5):
  plt.scatter (x_hat[y == k,0], x_hat [y == k,1],label ="Severity " + str(k), s = 3)

plt.title("US accidents")
plt.legend (loc="upper left")

plt.show ()

