'''
------------------------------------------------------------------------------------------------
                                       Machine Learning Module
------------------------------------------------------------------------------------------------

This module contains functions to create and analyse machine learning models of meds

'''
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from df_tools import drop_columns, to_numerical, fix_units, add_dummies, replace_nan_by_mean, divide_by_code, normalize_data
from pandas import DataFrame
from typing import Callable, Any, List
from gc import collect
import numpy as np
from os import mkdir
import json
from scipy.spatial.distance import euclidean
Numpy = np.ndarray
MODEL_PATH = r'G:\Data\prevention_iatrogenie_centrale_2\Models'

# prepare_df: takes an unprocessed DataFrame and applies several functions to it in order to prepare it
# to be used in machine learning models
def prepare_df(df: DataFrame, PRINT: bool = True, dummy: bool = True) -> DataFrame:
    if PRINT:
        print('Preparing and cleaning DataFrame for Clustering\n')
    df = df.rename({'TAILLE_UNITE': 'TAILLE_UNIT', 'POIDS_UNITE': 'POIDS_UNIT'}, axis=1)
    to_float = ['POIDS', 'TAILLE', 'BILIT', 'LEUCC', 'ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'VPM', 'RTCAMT', 
                'TQM', 'TQT', 'FIB', 'TCAT', 'QTE_PRESC']
    to_numerical(df, cols=to_float)
    if PRINT:
        print('Values converted to float\n')
    subfix = '_UNIT'
    fix_units(df, subfix=subfix, PRINT=False)
    if PRINT:
        print('Units fixed\n')
    df = drop_columns(df, '_UNITE')
    df = drop_columns(df, '_DATE')
    df = drop_columns(df, subfix)
    to_drop = ['DATE_INTERVENTION', 'P_IPP', 'P_ORD_C_NUM', 'P_LPM_C_NUM', 'P_LCP_C_NUM', 'P_LPO_C_NUM', 'NOM_COMMERCIAL', 'DCI', 'LIBELLE_MEDICAMENT',\
        'COMMENTAIRE_POSO', 'DATE_DEBUT', 'DATE_FIN', 'CODE_UCD', 'TYPE_FREQUENCE', 'HEURE_PRISE', 'MIN_PRISE', 'CONDITION', 'NON_QUOTIDIEN']
    df = df.drop(to_drop, axis=1)
    if PRINT:
        print('Columns dropped\n')
    df.dropna(subset=['CODE_ATC'], inplace=True)
    if PRINT:
        print('NaN values in CODE_ATC dropped\n')
    if dummy:
        to_dummies = ['SEXE', 'MOMENT']
        df = add_dummies(df, cols=to_dummies)
        SPREAD_AUCUN = False
        if SPREAD_AUCUN:
            df.loc[(df.MOMENT_aucun==1),['MOMENT_coucher', 'MOMENT_gouter',  'MOMENT_matin', 'MOMENT_matinée', 
                                         'MOMENT_midi', 'MOMENT_nuit', 'MOMENT_soir']] = 1
            df = df.drop('MOMENT_aucun', axis=1)
        if PRINT:
            print('Categorical data converted into dummies\n')    
    to_mean = ['AGE', 'POIDS', 'TAILLE', 'MDRD', 'CKDEPI', 'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC', 
                   'ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM', 'TCAT', 'RTCAMT', 
                   'TQM', 'TQT', 'TP', 'FIB']
    replace_nan_by_mean(df, to_mean)
    if PRINT:
        print('NaN values replaced by column\'s mean\n\n')
    return df


# get_centers_clusters: takes the points from a DataFrame and an array of predicted classes, then computes 
# and returns the centroids of every cluster. len(df) must be equal to len(labels)
def get_centers_clusters(df: DataFrame, labels: Numpy) -> Numpy:
    centers = []
    n_clust = np.max(labels)+1
    for n in range(n_clust):
        index_clust = np.where(labels==n)[0]
        center = df.loc[index_clust,:].mean().values
        centers.append(center)
    return np.array(centers)
    
# get_max_distances: takes the poins from a DataFrame, predicted labels from a model and the centers of its classes,
# returns the greatest distance of every point to its cluster center (some kind of "radius" of every class)
def get_max_distances(df: DataFrame, labels: Numpy, centers: Numpy) -> Numpy:
    n_clust = len(centers)
    max_dist = []
    for n in range(n_clust):
        X = df.values[np.where(labels==n)[0]]
        max_val = max((euclidean(X[j], centers[n]) for j in range(len(X))))
        max_dist.append(max_val)
    return np.array(max_dist)



def linear_regression(df: DataFrame, normalize: bool=True) -> dict():
    # linear_regression: Creates and fits a linear regression to predict "QTE_PRESC"
    # returns dict: {coef; incercept; mean; std; max; min}
    model = LinearRegression()
    mean = df.mean().tolist()
    std = df.std().tolist()
    if normalize:
        normalize_data(df)
        # scaler = MinMaxScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)
    data = df.drop(['QTE_PRESC'], axis=1)
    value = df['QTE_PRESC']
    model.fit(data, value)
    model_params = {'coef': model.coef_, 'intercept': model.intercept_, 'mean': mean, 'std': std,
                   'max': value.max(), 'min': value.min()}
    return model_params
    
    
def ridge_cv(df: DataFrame, normalize: bool=True) -> dict():
    # linear_regression: Creates and fits a linear regression to predict "QTE_PRESC"
    # returns dict: {coef; incercept; mean; std; max; min}
    model = RidgeCV()
    mean = df.mean().tolist()
    std = df.std().tolist()
    if normalize:
        normalize_data(df)
        # scaler = MinMaxScaler()
        # scaler.fit(data)
        # data = scaler.transform(data)
    data = df.drop(['QTE_PRESC'], axis=1)
    value = df['QTE_PRESC']
    model.fit(data, value)
    model_params = {'coef': model.coef_, 'intercept': model.intercept_, 'mean': mean, 'std': std,
                   'max': value.max(), 'min': value.min()}
    return model_params


def k_means(df: DataFrame, n_clusters=3) -> dict():
    # k_means: Creates and fits a KMeans model with a DataFrame and then returns it
    mean = df.mean().tolist()
    std = df.std().tolist()
    normalize_data(df)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)  
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    max_dist = get_max_distances(df, labels, centers)
    model_params = {'labels': labels, 'centers': centers, 'max_dist': max_dist, 'mean': mean, 'std': std}
    return model_params   

def meanshift(df: DataFrame, bandwidth=2) -> dict():
    normalize_data(df)
    kmeans = MeanShift(bandwidth=bandwidth)
    kmeans.fit(df)  
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    max_dist = get_max_distances(df, labels, centers)
    model_params = {'labels': labels, 'centers': centers, 'max_dist': max_dist}
    return model_params   


def hierarchical_clustering(df: DataFrame, linkage: str='average', distance_threshold: int=20, n_clusters=None) -> dict():
    # hierarchical_clustering : creates and fits a herarchical model with a DataFrame, 
    # a linkage and a distance_threshold and then returns it
    mean = df.mean().tolist()
    std = df.std().tolist()
    normalize_data(df)
    hiera_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, distance_threshold=distance_threshold)
    hiera_clustering.fit(df)
    labels = hiera_clustering.labels_
    centers = get_centers_clusters(df, labels)
    max_dist = get_max_distances(df, labels, centers)
    model_params = {'labels': labels, 'centers': centers, 'max_dist': max_dist, 'mean': mean, 'std': std}
    return model_params


'''
# radius_neighbors : creates and fits a RadiusNeighbors model with a DataFrame and a radius and then returns it
def radius_neighbors(df : DataFrame, rad=1.0) -> dict():
    normalize_data(df)
    neigh = RadiusNeighborsClassifier(radius=rad)
    labels = np.zeros(len(df))
    neigh.fit(df, labels)
    labels = neigh.predict(df).astype(int)
    centers = get_centers_clusters(df, labels)
    n_clust = len(centers)
    max_dist = get_max_distances(df, labels, centers)
    model_params = {'labels':labels, 'centers': centers, 'max_dist': np.array(max_dist)}
    return model_params
'''


def cluster_meds(df, group_by=None, ML_model=None, model_name=None, begin_at=None, min_obs=50, PRINT=True):
    '''
    cluster_meds : applies a given machine learning model to every drug in a processed DataFrame, then returns a
    dictionary where every key is a drug and its values are the dictionaries of the important parameters
    of the model (needed to create the warning alerting system of that model)
    '''
    MODEL_PATH = r'G:\Data\prevention_iatrogenie_centrale_2\Models' + '\\' + str(model_name)
    try:
        mkdir(MODEL_PATH)
    except:
        print('Folder already exists!')
    if group_by == None:
        group_by = ['CODE_ATC', 'VOIE_ADMIN', 'UNITE_PRESC']
    if ML_model == None:
        ML_model = k_means
    if PRINT:
        print('Clustering data grouped by:', str(group_by),'\n')
    groups = df.groupby(group_by)
    if PRINT:
        print('Data succesfully divided\n')
    models = dict()
    fails = 0
    for gp in groups:
        key = gp[0]
        if begin_at and key < begin_at:
            continue
        df_med = gp[1]
        n = len(df_med)
        if n < min_obs:
            del df_med
            collect()
            continue
        if PRINT:
            print(f'Clustering {key} with {n} points')
        df_med.drop(group_by,axis=1,inplace=True)
        df_med.reset_index(drop=True, inplace=True)
        try:
            model_params = ML_model(df_med)
            save_model(model_params, MODEL_PATH + '\\' + str(key).replace('/', '-') + '.json')
            del model_params
        except:            
            fails += 1
            print('Model creation failed for', str(key))       
        del df_med
        collect()
    if PRINT:
        print(str(fails), 'models failed')
    return


def save_model(model, path):
    for key in model.keys():
        if isinstance(model[key], np.ndarray):
            model[key] = model[key].tolist()
    with open(path, 'w') as file:
        json.dump(model, file)
    return  
    