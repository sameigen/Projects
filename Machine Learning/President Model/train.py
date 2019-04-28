#Sam Eigen
#presidential predictor algorithm
#4/5/2019

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn import utils

Features = ['fips', 'AGE775214', 'SEX255214', 'RHI125214', 'RHI225214', 'RHI325214',
'RHI425214', 'RHI525214', 'RHI625214', 'RHI725214', 'RHI825214', 'POP715213', 'POP645213',
'EDU685213', 'VET605213', 'LFE305213', 'HSG010214', 'HSG445213', 'HSG495213', 'HSD410213',
'HSD310213', 'INC910213', 'INC110213', 'PVY020213', 'SBO001207', 'LND110210', 'POP060210']

#take in 4 datasets: labels explained, county data,  presidential voting data in 2012, and voting data in 2016
featuresExplained = pd.read_csv("labels.csv")
CountyData = pd.read_csv("county data.csv")
VotingData = pd.read_csv("lineardata.csv")
FutureData = pd.read_csv("clintondata.csv")

S = CountyData.iloc[:,1:27].values
t = VotingData.values
future = FutureData.values

sc_S = preprocessing.StandardScaler()
sc_t = preprocessing.StandardScaler()
sc_2020 = preprocessing.StandardScaler()

S2 = sc_S.fit_transform(S)
t2 = sc_t.fit_transform(t)
futureElection = sc_2020.fit_transform(future)

#support vector regression
# clf = SVR(kernel='poly', degree=3, verbose=True) #gamma='auto'
clf = SVR(kernel='linear')
clf.fit(S2, t2)
initial = sc_t.inverse_transform(clf.predict(sc_S.transform(S)))
new = sc_2020.inverse_transform(clf.predict(sc_S.transform(S))) #the predictions
 #the predictions
print(new)
print(clf.score(S2, futureElection))
