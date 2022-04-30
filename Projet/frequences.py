'''
------------------------------------------------------------------------------------------------
                                       Frequences Module
------------------------------------------------------------------------------------------------

This module contains functions concerning the frequence columns.

'''

from os import chdir
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

import pandas as pd
from import_tools import import_subset
from df_tools import print_processbar
from time import time
from gc import collect
import csv
from pandas import DataFrame
from splitted_df_tools import mergeDF

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\securimed_filtered.csv"
PATH_EXIT = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data_"


def group_frequences(df: DataFrame, PRINT: bool = True):
    t_i = time()
    cols = ['P_ORD_C_NUM','CODE_ATC','VOIE_ADMIN','QTE_PRESC','UNITE_PRESC']
    cols2 = ['DATE_INTERVENTION'] + cols
    df2 = df[cols2].copy().sort_values(cols2).reset_index(drop = True) 
    df_list = df2.groupby(cols2)
    
    df2 = df.copy().sort_values(cols2).dropna(subset = ['CODE_ATC'], axis = 0).reset_index(drop = True)
    n = len(df2)
    k = 0
    for medicament in df_list:
        if PRINT:
            print_processbar(k, n, prefix = 'Filtering:', suffix = 'completed', length = 75)
            # print(f'Process: {k/n*100:.1f}%', end = '\r', flush = True)
        times_a_day = len(medicament[1])
        label = '//'.join([str(e) for e in medicament[0]])
        for i in range(k, k + times_a_day):
            df2.at[i, 'GROUP_BY'] = label            
        k += times_a_day    
    if PRINT:
        print(f'Duration de l\'algorithme: {time()-t_i:.3f} seconds')
    return df2
        

def group_to_csv(group, path: str, PRINT: bool = True, n: int = 100):
    with open(path, 'w', encoding = 'cp1252') as file:
        writer = csv.writer(file, delimiter = ';')        
        k = 0
        for med in group:
            if PRINT:
                print_processbar(k, n, prefix = 'Export to CSV:', suffix = 'completed', length = 75)   
            times_a_day = len(med[1])
            df = med[1].iloc[0]
            if times_a_day > 1:
                df['MOMENT'] = 'aucun'
            df['FREQUENCE'] = 24*60/times_a_day
            writer.writerow(list(df)[:-1])
            k += times_a_day
            
        
L = [[1, 146399], [146399, 292797], [292797, 439195], [439195, 585593], 
    [585593, 731991], [731991, 878389], [878389, 1024786], [1024786, 1171185], 
    [1171185, 1317584], [1317584, 1463981], [1463981, 1610378], [1610378, 1756777], 
    [1756777, 1903176], [1903176, 2049573], [2049573, 2195971], [2195971, 2342370], 
    [2342370, 2488766], [2488766, 2635165], [2635165, 2781562], [2781562, 2927972]]


def filter_frequences():
    k = 0
    for interval in L:
        print('Filtering DataFrame', str(k+1), 'of 20')        
        df = import_subset(PATH_DATA, from_to = interval)
        length = len(df)
        df = group_frequences(df)
        group_to_csv(df.groupby(['GROUP_BY']), PATH_EXIT + str(k) + '.csv', n = length)
        del df
        collect()
        k += 1

all_columns = ['DATE_INTERVENTION', 'P_IPP', 'P_ORD_C_NUM', 'P_LPM_C_NUM', 'P_LCP_C_NUM', 'P_LPO_C_NUM', 'DATE_DEBUT', \
    'DATE_FIN', 'CODE_UCD', 'CODE_ATC', 'NOM_COMMERCIAL', 'DCI', 'LIBELLE_MEDICAMENT', 'VOIE_ADMIN', 'QTE_PRESC', \
    'UNITE_PRESC', 'COMMENTAIRE_POSO', 'TYPE_FREQUENCE', 'MOMENT', 'HEURE_PRISE', 'MIN_PRISE', 'FREQUENCE', 'CONDITION', \
    'NON_QUOTIDIEN', 'SEXE', 'AGE', 'POIDS', 'POIDS_UNITE', 'POIDS_DATE', 'TAILLE', 'TAILLE_UNITE', 'TAILLE_DATE', 'MDRD', \
    'MDRD_UNITE', 'MDRD_DATE', 'CKDEPI', 'CKDEPI_UNITE', 'CKDEPI_DATE', 'BILIT', 'BILIT_UNITE', 'BILIT_DATE', 'PAL', 'PAL_UNITE', \
    'PAL_DATE', 'TGO', 'TGO_UNITE', 'TGO_DATE', 'TGP', 'TGP_UNITE', 'TGP_DATE', 'LEUCC', 'LEUCC_UNITE', 'LEUCC_DATE', 'ERYTH', \
    'ERYTH_UNITE', 'ERYTH_DATE', 'HHB', 'HHB_UNITE', 'HHB_DATE', 'HTE', 'HTE_UNITE', 'HTE_DATE', 'VGM', 'VGM_UNITE', 'VGM_DATE', \
    'TCMH', 'TCMH_UNITE', 'TCMH_DATE', 'CCMH', 'CCMH_UNITE', 'CCMH_DATE', 'NPLAQ', 'NPLAQ_UNITE', 'NPLAQ_DATE', 'VPM', 'VPM_UNITE',\
    'VPM_DATE', 'TCAM', 'TCAM_UNITE', 'TCAM_DATE', 'TCAT', 'TCAT_UNITE', 'TCAT_DATE', 'RTCAMT', 'RTCAMT_UNITE', 'RTCAMT_DATE', 'TQM',\
    'TQM_UNITE', 'TQM_DATE', 'TQT', 'TQT_UNITE', 'TQT_DATE', 'RTQMT', 'RTQMT_UNITE', 'RTQMT_DATE', 'TP', 'TP_UNITE', 'TP_DATE', 'FIB',\
    'FIB_UNITE', 'FIB_DATE']

data_path = r"G:\Data\prevention_iatrogenie_centrale_2\grouped_data_"

def merge_csv() -> DataFrame:
    dflist = []
    for i in range(20):
        path = data_path + str(i) + '.csv'
        df = pd.read_csv(path, names=all_columns, encoding='cp1252', sep=';')
        dflist.append(df)
        print(f'DataFrame #{i} imported')
    df = mergeDF(dflist)
    return df

