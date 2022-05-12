import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
from scipy import stats
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

df = pd.read_csv('US_Accidents_Dec21_updated.csv')

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Hour']=df['Start_Time'].dt.hour
td='Time_Duration(min)'
df[td]=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))

features = ['Severity','Distance(mi)','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Crossing','Junction','Traffic_Signal','Hour','Time_Duration(min)' ]
df['Crossing'] = df['Crossing'].apply(lambda x : 0.0 if(x == 'false') else 1.0)
df['Junction'] = df['Junction'].apply(lambda x : 0.0 if(x == 'false') else 1.0)
df['Traffic_Signal'] = df['Traffic_Signal'].apply(lambda x : 0.0 if(x == 'false') else 1.0)
df_feat=df[features].copy()
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
x,y=shuffle(x,y)
x=np.asarray(x)
y=np.asarray(y)
x_train , x_test , y_train , y_test = train_test_split (x, y, test_size = 0.2)

def kernelKNN(x, X, Y, nn):
	d = LA.norm(x -X, axis=1) 
	idx = np.argsort(d)
	y_hat , _ = stats.mode(Y[idx[0:nn]]) 
	return y_hat[0]

def KNN(X_train, X_test, Y_train, nn):
	X_train = X_train[np.random.choice(len(X_train), 100, replace=False)]
	Y_hat = []
	for x in range(len(X_test)):
		y_hat = kernelKNN(x, X_train, Y_train, nn)
		Y_hat.append(y_hat)
	return np.asarray(Y_hat)

Y_hat = KNN(x_train, x_test, y_train, nn = 3)

acc = np.sum(Y_hat == y_test)/len(Y_hat)

print('KNN Accuracy = %0.4f'%(acc))

confmat = confusion_matrix(y_true=y_test, y_pred=Y_hat)

index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - K Nearest Neighbor")
plt.show()
