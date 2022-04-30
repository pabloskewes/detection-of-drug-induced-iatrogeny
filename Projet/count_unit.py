from import_tools import import_splitted
from splitted_df_tools import process_df, merge_dicts
import pandas as pd
from df_tools import divide_by_code, drop_columns
from gc import collect

PATH_ECHANTILLON = "G:\Data\prevention_iatrogenie_centrale_2\echantillon.csv"
PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\securimed2.csv"

code = ["DCI"]

df_list = import_splitted(PATH_DATA, 5)
df_list = process_df(df_list, lambda df : drop_columns(df, "_UNITE"))
df_list = process_df(df_list, lambda df : drop_columns(df, "_DATE"))
df_dict = process_df(df_list, lambda df : divide_by_code(df, code))
del df_list
collect()
df_dict = merge_dicts(df_dict)
print(df_dict)
#dict_final = merge_dicts(df_dict, )



