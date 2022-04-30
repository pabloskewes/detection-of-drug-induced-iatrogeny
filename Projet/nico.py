from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from import_tools import import_data
from machine_learning import k_means, linear_regression
import json



# def closest(num, L):
    # dist = [np.absolute(num - value) for value in L]
    # return L[dist.index(min(dist))]

DOLIPRANE = "G:\Data\prevention_iatrogenie_centrale_2\doliprane.csv"
df = import_data(DOLIPRANE).drop(['Unnamed: 0'], axis = 1)
df.loc[:, 'RTQMT'] = df['RTQMT'].fillna(value=df['RTQMT'].mean())

data = df.drop(['QTE_PRESC'], axis = 1)
qte = df['QTE_PRESC']

scaler = MinMaxScaler()
scaler.fit(data)
data2 = scaler.transform(data)
X_train, X_test, y_train, y_test = train_test_split(data2, qte, test_size = .2)





        
