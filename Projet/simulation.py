import numpy as np
import pandas as pd
from warning_system import warning_system
from import_tools import import_data, import_random_subset
import json
from os import walk
from gc import collect
from ast import literal_eval
from machine_learning import save_model

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"
PROCESSED_DATA = "G:\Data\prevention_iatrogenie_centrale_2\processed_data.csv"

def random_sample(df, sample_size: int =2000, size_true: float =0.5):
    true_values = int(sample_size*size_true)
    lr_names = next(walk(r'G:\Data\prevention_iatrogenie_centrale_2\Models\linear_regression'), (None, None, []))[2]
    km_names = next(walk(r'G:\Data\prevention_iatrogenie_centrale_2\Models\k_means'), (None, None, []))[2]
    hc_names = next(walk(r'G:\Data\prevention_iatrogenie_centrale_2\Models\hierarchical_clustering'), (None, None, []))[2]
    lr_names = [literal_eval(name[:-5].replace('-', '/')) for name in lr_names]
    km_names = [literal_eval(name[:-5].replace('-', '/')) for name in km_names]
    hc_names = [literal_eval(name[:-5].replace('-', '/')) for name in hc_names]
    df2 = pd.DataFrame()
    df2.loc[:, 'keys'] = df[['CODE_ATC', 'VOIE_ADMIN', 'UNITE_PRESC']].apply(tuple, axis=1)
    index = []
    k = 0
    while k < sample_size:
        r = np.random.randint(0, len(df)+1)
        if r in index:
            continue
        key = df2.at[r, 'keys']
        if all([key in lr_names, key in km_names, key in hc_names]):
            index.append(r)
            k += 1          
    df_sample = df.loc[index]
    df_true = df_sample.sample(n=true_values)
    df_false = df_sample.loc[~df_sample.index.isin(df_true.index)]
    for col in df_false.columns:
        df_false.loc[:, col] = np.random.permutation(df_false[col].values)
    df_false.loc[:, 'label'] = 0
    df_true.loc[:, 'label'] = 1
    return pd.concat([df_true, df_false]).sort_index().reset_index(drop=True)


def test():
    df = import_random_subset(PATH_DATA, PRINT=False)
    df_sample = random_sample(df)
    print(f'Imported a {len(df_sample)} sample')
    X = df_sample.drop(['label'], axis=1)
    y = df_sample['label']
    y_y_pred = []
    index = []
    for k in range(len(X)):
        # try:
        prediction = 0 if warning_system(X.iloc[k,:]) else 1
        y_pred.append(prediction)
        index.append(k)
        # except:
            # print("Didn't work!")        
    y_pred = np.array(y_pred)
    y = y.loc[index]
    print(f'{len(y) - len(index)} models falied to be loaded')
    conf_matrix = pd.crosstab(y, y_pred, colnames=['Real Class'], rownames=['Predicted Class'])
    print(conf_matrix)
    

def code_atc_patients(df):
    PATH = r'G:\Data\prevention_iatrogenie_centrale_2\Patients'
    patiens = df.groupby('P_IPP')
    for gp in patiens:
        id_pat = gp[0]
        df_pat = gp[1]
        codes = {id_pat: df_pat['CODE_ATC'].tolist()}
        path = PATH + '\\' + str(id_pat) + '.json'
        print('Saving data in ',path)
        try:
            save_model(codes, path)
            print(f"Saved {id_pat} with {df_pat['CODE_ATC'].tolist()}")
        except:
            print('Error: patient data not saved')




