'''
------------------------------------------------------------------------------------------------
                                          Splitted DF Module
------------------------------------------------------------------------------------------------

This module containts useful functions for treating splitted dataframes.

                                          f: df**n ---> X

These functions work for splitted dfs, if you are working with a single df you could use
the functions like this: dflist = [df]

General parameters:
    dflist   -> splitted dataframe (list of dfs imported with "import_splitted")
    PRINT    -> if True, prints info about the imported data

mergeDF:
    -> You can try to merge a dflist with this function, it may not work. If it works, you can 
    still use every function for dflist like [df_merged].



'''
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from typing import List, Any, Callable
import csv

from df_tools import column_count

# process_df: list(df) , function(df -> A)  -> list(A)
# applies a funcion to every dataframe in a list
def process_df(dflist: List[DataFrame], function: Callable[[DataFrame], Any]) -> List[Any]:
    return [ function(dflist[i]) for i in range(len(dflist)) ]

# mergeDF: list(df) -> df (if it works)
# tries to merge a dflist into a unique df and if it works it returns it. 
# ff it doesn't work, it returns the same df list.
def mergeDF(dflist: list[DataFrame], PRINT: bool = True) -> DataFrame:
    try:
        dftot = pd.concat(dflist,ignore_index=True)
    except:
        print('Not enought memory to merge df list into a df.')
        dftot = dflist
    else:
        if PRINT:
            print('Dataframe succesfully merged.')
    finally:
        return dftot

# merge_dicts: takes a list of dicts of dfs, returns a single dict with
# the concatenated dfs for the indicated meds (keys). it also drops nan values
def merge_dicts(dict_list: List[dict[str,DataFrame]]) -> dict[str,DataFrame]:
    D = {}
    set_keys = set()
    for dict in dict_list:
        set_keys.update(list(dict.keys()))
    for key in set_keys:
        dflist_med = [dict[key] for dict in dict_list if key in dict.keys()]
        D[key] = mergeDF(dflist_med,PRINT=False)
    return D


# total_value_counts: list(df), str -> series
# returns "value_counts" of the union of dataframes in a list. i.e: counts the values for every df in the list.
def total_value_counts(dflist: list[DataFrame], col_name: str) -> Series:
    function = lambda X: column_count(X,col_name)
    series_list = process_df(dflist, function)
    merged = pd.Series()
    for i in range(len(dflist)):
        merged = merged.combine(series_list[i], lambda x,y: x+y, fill_value = 0)
    merged = merged.astype(int)
    return merged

# count_everything: list(df) bool -> dict
# counts values for every columns and saves it into a dictionary
def count_everything(dflist: list[DataFrame], PRINT = True) -> dict:
    counters = []
    for col in dflist[0].columns:
        if PRINT:
            print('counting values in',col)
        counters.append(total_value_counts(dflist,col))
    cols = dflist[0].columns.tolist()
    D = dict(zip(cols,counters))
    if PRINT:
        print('printing number of different values for every columns:')
        for key in D:
            print(key,':',len(D[key]))
    return D



# write_csv: list(df) str str str -> None
# writes a list of DataFrames into a .csv file
def write_csv(df_list: list[DataFrame], path: str, sep = ',', encoding = 'UTF-8') -> None:
    with open(path, 'w', encoding = encoding) as file:
        writer = csv.writer(file, delimiter = sep)
        writer.writerow(list(df_list[0].columns))
        for df in df_list:
            for i in range(len(df)):
                writer.writerow(list(df.loc[i, :]))

