'''
------------------------------------------------------------------------------------------------
                                       Système d'Alerte
------------------------------------------------------------------------------------------------

This module contains functions to create the warning system for medical prescriptions

'''
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from df_tools import drop_columns, to_numerical, fix_units, add_dummies, replace_nan_by_mean, divide_by_code
from pandas import DataFrame, Series
from typing import Callable, Any, Tuple, List
import numpy as np
from machine_learning import prepare_df
from scipy.spatial.distance import euclidean
from gensim.models import Word2Vec
from import_tools import import_json


def prepare_line(presc_line: Series, group_by: List[str] = None, replace_nan_dict = None, replace_nan_funct = None ) -> Tuple[List[str], Series]:
    if group_by == None:
        group_by = ['CODE_ATC', 'VOIE_ADMIN', 'UNITE_PRESC']
    keys = tuple(presc_line[group_by].values)
    assert replace_nan_dict == None or replace_nan_funct == None
    if replace_nan_dict == None:
        replace_nan_dict = {'AGE': 69.36, 'POIDS': 150.86, 'TAILLE': 128.33, 'MDRD': 67.58, 'CKDEPI': 67.34,'BILIT': 8.14, 'PAL': 117.85, 'TGO': 35.74,
                           'TGP': 29.71, 'LEUCC': 9.46, 'ERYTH': 4.01, 'HHB': 11.85, 'HTE': 36.58, 'VGM': 91.81, 'TCMH': 29.72, 'CCMH': 32.35, 'NPLAQ': 273.22,
                          'VPM': 10.39, 'TCAM': 37.81, 'TCAT': 31.03, 'RTCAMT': 1.22, 'TQM': 17.41, 'TQT': 13.66, 'TP': 79.85, 'FIB': 4.7}
    if replace_nan_funct == None:
        replace_nan_fucnt = lambda x: x
    for key in replace_nan_dict:
        presc_line[key] = replace_nan_dict[key]
    def replace_nan(x):
        if x != x:
            return replace_nan_fucnt(x)
        else:
            return x
    presc_line = presc_line.map(replace_nan)
    df_line = presc_line.to_frame().T.reset_index(drop=True)
    df_line = prepare_df(df_line, dummy=False, PRINT=False)
    df_line.loc[1] = {'SEXE': 'M'}
    df_line.loc[2] = {'SEXE': 'F'}
    moments = ['aucun', 'matin', 'soir', 'midi', 'coucher', 'matinée', 'gouter', 'nuit']
    for i in range(len(moments)):
        df_line.loc[i+3] = {'MOMENT': moments[i]}
    df_line = add_dummies(df_line, cols=['SEXE','MOMENT'])
    presc_line = df_line.loc[0].copy()
    return keys, presc_line.drop(group_by + ['RTQMT'])


def normalize_line(line, mean, std):
    return (line - mean)/std

def warning_lr(line: Series, model_params: dict) -> bool:
    '''
    warning_lr: Series dict -> bool
    Predicts the prescribed dose, then compare it with the original value. 
    Launch the alert if the difference between the values is greater than 
    the standard deviation.
    '''
    std = model_params['std'].values
    mean = model_params['mean'].values
    point = normalize_line(line.values, mean, std)
    qte = point[0]
    point = point[1:]
    coef = model_params['coef'].values
    intercept = model_params['intercept'].values[0]
    prediction = intercept + np.sum(np.multiply(point, coef))
    return np.abs(qte - prediction) > std[0]


 
def warning_cluster(line: Series, model_params: dict, percentage_coef=1) -> bool:
    '''
    We determine which center is the closest to the input point.
    Then we calculate the distance between the input point and its associated center.
    We calculate the maximum_distance, which is the distance between the associated center and the farest 
    point belonging to the same cluster.
    If the initial distance is superior to the maximum_distance * percentage_coef, we return "True" which 
    means that we send an alert.
    '''
    mean = model_params['mean']
    std = model_params['std']
    point = normalize_line(line.values, mean, std)
    normalize_line
    centers = model_params['centers']
    distance_to_clusters = [euclidean(center, point) for center in centers]
    index_center = np.argmax(distance_to_clusters)
    center = centers[index_center]
    distance = euclidean(center, point)
    maximum_distance = model_params['max_dist'][index_center]
    return distance > maximum_distance*percentage_coef

    
def warning_word2vec(presc_line: Series) -> Tuple[bool, str]:
    '''
    We observe the contexte for the ATC CODE. 
    '''
    patient = presc_line['P_IPP']
    code = presc_line['CODE_ATC']
    model = Word2Vec.load("word2vecComplete.model")
    context = [atc[0] for atc in model.wv.most_similar(code)]
    PATH = r'G:\Data\prevention_iatrogenie_centrale_2\Patients' + '\\' + str(patient) + '.json'
    antecedents = import_json(PATH)
    ordonnance = set(antecedents[patients])
    for code in ordonnance:
        if code not in context:
            return False, code
    return True, None


def warning_system(presc_line: Series, use_models = None, mode=all) -> bool:
    keys, line = prepare_line(presc_line)
    PATH_LR = r'G:\Data\prevention_iatrogenie_centrale_2\Models\linear_regression'
    PATH_RIDGE = r'G:\Data\prevention_iatrogenie_centrale_2\Models\ridge_cv'
    PATH_KMEANS = r'G:\Data\prevention_iatrogenie_centrale_2\Models\k_means'
    PATH_HIERARCHICAL = r'G:\Data\prevention_iatrogenie_centrale_2\Models\hierarchical_clustering'
    keys_str = str(keys).replace('/','-')+'.json'
    # Regression logistique
    lr_params = import_json(PATH_LR + '\\'+ keys_str)
    bool_lr = warning_lr(line,lr_params)
    # Ridge CV
    ridge_params = import_json(PATH_RIDGE + '\\'+ keys_str)
    bool_ridge = warning_lr(line,ridge_params)
    # K means
    kmeans_params = import_json(PATH_KMEANS + '\\'+ keys_str)
    bool_kmeans = warning_cluster(line,kmeans_params)
    # Hierarchical
    hierarchical_params = import_json(PATH_HIERARCHICAL + '\\'+ keys_str)
    bool_hierarchical = warning_cluster(line,hierarchical_params)
    
    # Word2vec
    #bool_word2vec, _ = warning_word2vec(line,word2vec_params)
    bool_word2vec = False
    
    if use_models == None:
        use_models = {'lr': True, 'km': True, 'hc': True}
    use = np.array(list(use_models.values()))
    bools_index = np.where(use==True)[0]
    bool_list = np.array([bool_lr, bool_kmeans, bool_hierarchical])
    bool_list = bool_list[bools_index]
    
    return mode(bool_list) or bool_word2vec
   
