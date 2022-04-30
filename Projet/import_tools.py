'''
------------------------------------------------------------------------------------------------
                                    Importing dataframes module
------------------------------------------------------------------------------------------------

This module containts useful functions for data importing.

The following is a detailed description of its parameters

Every funcion has these parameters:
    filename -> path where file (in csv) is located
    PRINT    -> if True, prints info about the imported data

import_subset:
    from_to  -> list of lenght 2 that indicates in which range we want to import the data
    ex: from_to = [10,30] imports data from line 10 to 29 (2nd coordonate it's excluded)

import_random_subset:
    size        -> sets size of random sample to import
    random_seed -> sets a random state

import_splitted:
    n_samples   -> sets in how many parts the data will be splitted
    ex: if L = range(9) and n_samples = 3 => splitted data would be: [[0,1,2],[3,4,5],[6,7,8,9]]
    randomize   -> if true, then the splitted data will be randomly distributed
    random_seed -> sets a random state (only makes sense if "randomize" is True)

'''

import pandas as pd
from pandas import DataFrame, Series
import random
import json
from typing import List


# import_data: str -> df
# imports all data in a dataframe
def import_data(filename: str) -> DataFrame:
    return pd.read_csv(filename, encoding = 'cp1252', sep = ';')

# import_subset: str list(int,int) bool -> df
# imports data within a given range, returns it in dataframe
def import_subset(filename: str, from_to: List = [1,100000], PRINT: bool = True) -> DataFrame:
    if PRINT:
        print('Reading file from',filename)
    n = sum(1 for line in open(filename)) - 1
    assert from_to[1] <= n
    if PRINT:
        print(n,'lines ready to be imported')
    skip = list(set(range(1,n+1)) - set(range(from_to[0],from_to[1])))
    df = pd.read_csv(filename, encoding = 'cp1252', sep = ';', skiprows = skip)
    if PRINT:
        print('Lines saved from',from_to[0],'to',from_to[1]-1)
        print(len(df), 'lines loaded in dataframe')
    return df


# import_random_subset: str int int bool -> df
# imports random subset of given size of data and returns dataframe
def import_random_subset(filename: str, size: int = 100000, random_seed: int = 0, PRINT: bool = True) -> DataFrame:
    if PRINT:
        print('Reading file from',filename)
    random.seed(random_seed)
    n = sum(1 for line in open(filename)) - 1
    if PRINT:
        print(n,'lines ready to be imported')
    skip = sorted(random.sample(range(1,n+1),n-size))
    df = pd.read_csv(filename, encoding = 'cp1252', sep = ';', skiprows = skip)
    if PRINT:
            print(len(df), 'lines loaded in dataframe')
    return df

# import_splitted_: str int bool -> list(df)
# imports all data separated in n dataframes 
def import_splitted(filename: str, n_samples: int = 4, PRINT: bool = True) -> List[DataFrame]:
    if PRINT:
        print('Reading file from',filename)
    n = sum(1 for line in open(filename)) - 1                           
    if PRINT:
        print(n,'lines ready to be imported')
    s = n//n_samples

    create_gen = lambda i: (j + 1 for j in range(0,n) if j not in range(s*i, s*(i+1)))
    gen_list = [create_gen(i) for i in range(n_samples-1)]
    last_gen = (j + 1 for j in range(0,n) if j not in range(s*(n_samples-1), n)) 
    gen_list.append(last_gen)

    dataframe_list = []
    for i in range(n_samples):
        df = pd.read_csv(filename, encoding = 'cp1252', sep = ';', skiprows = gen_list[i])
        dataframe_list.append(df)
        if PRINT:
            print(len(df), 'lines loaded in dataframe_'+str(i))
    return dataframe_list


# import_json: import json in dataframe
def import_json(filename: str) -> dict[str,Series]:
    f = open(filename)
    data = json.load(f)
    D = {}
    for key in data:
        D[key] =  Series(data[key])
    return D


# ----------------------------------------------------------------------------------------------------------------

# import_splitted_2: str int bool bool int -> list(df)
# imports all data separated in n dataframes in the same order or randomly (at choice changing "randomize" value)
def import_splitted_2(filename: str, n_samples: int = 4, randomize: bool = False, PRINT: bool = True, random_seed: int = 0) -> List[DataFrame]:
    if PRINT:
        print('Reading file from',filename)
    random.seed(random_seed)
    n = sum(1 for line in open(filename)) - 1                           
    if PRINT:
        print(n,'lines ready to be imported')
    s = n//n_samples
    r = n % n_samples
    df_sizes = [s for i in range(n_samples-1)]
    df_sizes.append(s+r)
    assert sum(df_sizes) == n
    index_dispo = range(1,n+1)
    if randomize:
        skip_list = []
        for i in range(n_samples):
            randlist = random.sample(index_dispo,df_sizes[i])
            to_skip = sorted(list(set(range(1,n+1)) - set(randlist)))
            skip_list.append(to_skip)
            index_dispo = list(set(index_dispo) - set(randlist))
        if PRINT:
            print('Random index created')
    else:
        split_test = []
        for i in range(1,n_samples):
            split_test.append(i*s)
        save = [index_dispo[i:j] for i,j in zip([0]+split_test,split_test + [None])]
        skip_list = []
        for i in range(n_samples):
            to_skip = sorted(list(set(index_dispo) - set(save[i])))
            skip_list.append(to_skip)
    if PRINT:
        print('File succesfully splitted in',n_samples)
    dataframe_list = []
    for i in range(n_samples):
        df = pd.read_csv(filename, encoding = 'cp1252', sep = ';', skiprows = skip_list[i])
        dataframe_list.append(df)
        if PRINT:
            print(len(df), 'lines loaded in dataframe_'+str(i))
    return dataframe_list
