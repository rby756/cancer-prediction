from pyexpat import model
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import pickle

data=pd.read_csv('cancer.csv')


drop_list=['perimeter_mean','radius_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst','concave points_worst','compactness_se','concave points_se','texture_worst','area_worst','id','Unnamed: 32']

data.drop(drop_list,axis=1,inplace=True)


data['diagnosis']=data['diagnosis'].replace(['B','M'],[0,1]).astype('category')


x=data.drop('diagnosis', axis=1)
y=data['diagnosis']


model=AdaBoostClassifier()
model.fit(x,y)

#saving model to disk
pickle.dump(model,open('model.pkl','wb'))

#read the model to comapare
model=pickle.load(open('model.pkl','rb'))


print(x.shape)
print(x.columns)