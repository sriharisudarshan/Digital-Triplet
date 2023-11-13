# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:41:42 2021

@author: sriha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ds=pd.read_csv('sensor.csv')
ds=ds.drop(['timestamp','Unnamed: 0','sensor_15'],axis=1)

ds.replace([np.inf, -np.inf], np.nan, inplace=True) 
ds=ds.fillna(ds.mean())
ds.shape
x=ds.iloc[:,0:51].values

ds['machine_status'].unique()

#label_encoder
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
ds['machine_status']= label_encoder.fit_transform(ds['machine_status']) 
y=ds.iloc[:,-1].values

#splitting data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

ds.plot(subplots =True, sharex = True, figsize = (20,50))


#logistic regression
from sklearn.linear_model import LogisticRegression
c= LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)

#prediction
ypred=c.predict(xtest)

