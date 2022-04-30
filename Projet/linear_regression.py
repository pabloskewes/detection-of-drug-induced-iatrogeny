'''
------------------------------------------------------------------------------------------------
                                       Linear Regression Module
------------------------------------------------------------------------------------------------

This module contains functions to create and analyse the linear regression of meds

'''

from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"

import pandas as pd
import numpy as np
from import_tools import import_splitted
from splitted_df_tools import process_df, mergeDF
from sklearn.linear_model import LinearRegression
from machine_learning import prepare_df
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df_list = import_splitted(PATH_DATA, n_samples=5)

df_list = process_df(df_list, prepare_df)
df = mergeDF(df_list)
del df_list
collect()

df = df[df['CODE_ATC'] == 'N02BE01']
df = df[df['VOIE_ADMIN'] == 'ORALE']
df = df[df['UNITE_PRESC'] == 'mg']

df.drop(['CODE_ATC', 'VOIE_ADMIN', 'UNITE_PRESC'], axis = 1, inplace = True)


def linear_regression(df: pd.DataFrame, normalize: bool = True):
    model = LinearRegression()
    data = df.drop(['QTE_PRESC'], axis = 1)
    value = df['QTE_PRESC']
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(data)
        data = scaler.transform(data)
    model.fit(data, value)
    return model
    