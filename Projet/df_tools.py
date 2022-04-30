'''
------------------------------------------------------------------------------------------------
                                       DF_tools Module
------------------------------------------------------------------------------------------------

This module contains useful functions for treating dataframes.

                                      f: df ---> X

*** L'idée est que vous créez les fonctions pour traiter les df ici (n'importe quoi, l'important
es qu'elle prend un seul df. Après, grace aux fonctions du module splitted_df_tools, vous pouvez
apliquer vos fonctions á plusieurs df (une liste de df)

'''

from pandas import DataFrame, Series
from typing import Any, Union, List
from gc import collect
import pandas as pd
from time import time
import os
from sklearn.preprocessing import StandardScaler




def column_count(df: DataFrame, col_name: str) -> Series:
    # column_count: df  str  ->  series
    # applies value_counts to df
    return df[col_name].value_counts()



def filter_eq(df: DataFrame, col_name: str, val: Any, show_all: bool = True) -> Union[DataFrame,Series]:
    # filter_eq: df  str  any  ->  df / series
    # saves values of a df th
    if show_all:
        return df[df[col]==val]
    else:
        return df[col][df[col]==val]



def fix_units(df: DataFrame, convert_dict = dict(), subfix = '_UNITE', PRINT: bool = True) -> None:
    '''
    fix_units: fixes units of DataFrame by a dictonnary where keys are columns to fix 
    and values are dicts of unit convertion
    example of dict: {'TAILLE': {'g': ('kg',1/1000)}, 'POIDS': {'cm': ('m',1/100}}
    '''
    if convert_dict == {}:
        convert_dict = {'POIDS': {'g': ('kg',1/1000)}, 'TAILLE': {'cm': ('m',1/100)}}
    for col in list(convert_dict.keys()):
        if PRINT:
            print(f'Converting units of {col}')
        for unit in list(convert_dict[col].keys()):
            new_unit = convert_dict[col][unit][0]
            factor = convert_dict[col][unit][1]
            unit_col = col + subfix
            if PRINT:
                print(f'{unit} -> {new_unit} : multiplying by {factor}')
            df.loc[(df[unit_col] == unit), col] *= factor
            df.loc[(df[unit_col] == unit), unit_col] = new_unit


def divide_by_code(df : DataFrame, codes : List[str], reset_index: bool=False, min_observations: int=30) -> dict():
    '''
    Divide a DF by a code(s), returning a dict with
    key: a code, value: DF with column code dropped
    Return the dict
    '''
    gp = df.groupby(codes)
    dfs = dict()
    for i in gp:
        if len(i[1]) < min_observations:
            continue
        dfs[i[0]] = i[1].drop(codes, axis=1)
        if reset_index:
            dfs[i[0]].reset_index(drop = True, inplace = True)
    return dfs



def drop_columns(df: DataFrame, filter_string: str) -> DataFrame:
    '''
    Drop columns from a DataFrame by a filter string
    example: "_UNITE" or "_DATE"
    returns df with columns dropped
    '''
    filter = [col for col in df.columns if filter_string in col]
    return df.drop(filter, axis = 1)

    

def to_numerical(df: DataFrame, cols: List[str]=None) -> None:
    '''
    to_numerical: transform elements in DataFrame to float for a given list of columns
    exemple: 'HTE' or 'BILIT'
    works inplace (returns None)
    '''
    if cols == None:
        cols = ['POIDS', 'TAILLE', 'BILIT', 'LEUCC', 'ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'VPM', 'RTCAMT', 
                'TQM', 'TQT', 'FIB', 'TCAT', 'QTE_PRESC']
    df.loc[:, cols] = df[cols].applymap(str, na_action='ignore')
    df.loc[:, cols] = df[cols].applymap(lambda x: x.replace(',','.'), na_action='ignore')
    df.loc[:, cols] = df[cols].applymap(float, na_action='ignore')
    return


def add_dummies(df: DataFrame, cols: List[str]) -> DataFrame:
    '''
    add_dummies: join dummies of a given list of columns to a DataFrame then drops the
    original columns. Works inplace
    '''
    df = df.join(pd.get_dummies(df[cols]))
    df = df.drop(cols, axis=1)
    return df
 

def replace_nan_by_mean(df: DataFrame, cols: List[str]):
    # replace_nan_by_mean : replace de NaN values with de mean of the column, for all colums in cols   
    df.loc[:, cols] = df[cols].fillna(value=df[cols].mean()) 

def normalize_data(df: DataFrame, to_normalize=None):
    # normalize the data of the DataFrame
    if to_normalize == None:
        to_normalize = ['QTE_PRESC','FREQUENCE','AGE', 'POIDS', 'TAILLE', 'MDRD', 'CKDEPI', 'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC', 
                   'ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM', 'TCAT', 'RTCAMT', 
                   'TQM', 'TQT', 'TP', 'FIB']
    scaler = StandardScaler()
    df.loc[:,to_normalize] = scaler.fit_transform(df.loc[:,to_normalize])
    return 

def group_frequences(df: DataFrame, PRINT: bool = True) -> DataFrame:
    t_i = time()
    cols = ['P_ORD_C_NUM','CODE_ATC','VOIE_ADMIN','QTE_PRESC','UNITE_PRESC']
    cols2 = ['DATE_INTERVENTION'] + cols
    df2 = df[cols2].copy().sort_values(cols2).reset_index(drop = True) 
    df2.drop(['DATE_INTERVENTION'], axis = 1)
    df_list = df2.groupby(cols2)
    
    df2 = df.copy().sort_values(cols2).reset_index(drop = True)   
    n = len(df2)
    k = 0
    for medicament in df_list:
        if PRINT:
            print_processbar(k, n, prefix = 'Progress:', suffix = 'completed', length = 75)
            # print(f'Process: {k/n*100:.1f}%', end = '\r', flush = True)
        times_a_day = len(medicament[1])
        if times_a_day > 1:
            df2.at[k,'MOMENT'] = 'AUCUN'
        df2.at[k, 'FREQUENCE'] = 24*60/times_a_day
        df2.drop(range(k+1, k + times_a_day), axis = 0, inplace = True)
        k += times_a_day
    df2.reset_index(drop = True, inplace = True)
    if PRINT:
        print(f'Duration de l\'algorithme: {time()-t_i:.3f} seconds')
    return df2


def print_processbar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '\u2588', printEnd = '\r'):
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-'*(length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', flush = True, end = printEnd)
    
    if iteration == total:
        print(f'', flush = True, end = printEnd)
    return


