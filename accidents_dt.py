import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('US_Accidents_Dec21_updated.csv')

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
df['Hour']=df['Start_Time'].dt.hour
td='Time_Duration(min)'
df[td]=round((df['End_Time']-df['Start_Time'])/np.timedelta64(1,'m'))


features = ['Severity','Distance(mi)','Temperature(F)','Humidity(%)','Pressure(in)','Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Hour','Time_Duration(min)' ]
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
x_train , x_test , y_train , y_test = train_test_split (x, y, test_size = 0.2)

tree_clf = DecisionTreeClassifier(max_depth=6)
accidents_tree = tree_clf.fit(x_train,y_train)

y_pred = accidents_tree.predict(x_test)
acc = np.sum(y_pred == y_test)/len(y_pred)

print('Decision Tree accuracy = %0.4f'%(acc))

plt.figure(figsize=(14,8))
plot_tree(decision_tree=accidents_tree, feature_names=x.columns,filled=True,fontsize=7)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix - Decision tree")
plt.show()