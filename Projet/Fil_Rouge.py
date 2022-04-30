from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"
DOLIPRANE = "G:\Data\prevention_iatrogenie_centrale_2\doliprane.csv"
PROCESSED_DATA = "G:\Data\prevention_iatrogenie_centrale_2\processed_data.csv"

from import_tools import import_splitted, import_data, import_subset

df_list = import_splitted(PATH_DATA, n_samples=5)

from splitted_df_tools import process_df, mergeDF

from machine_learning import prepare_df
df_list = process_df(df_list, prepare_df)
df = mergeDF(df_list)
del df_list
collect()



# --------------------------------------------------------------------------------------
from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"
DOLIPRANE = "G:\Data\prevention_iatrogenie_centrale_2\doliprane.csv"

from import_tools import import_subset
from warning_system import prepare_line
df = import_subset(PATH_DATA)
to_numerical(df)
to_mean = ['AGE', 'POIDS', 'TAILLE', 'MDRD', 'CKDEPI', 'BILIT', 'PAL', 'TGO', 'TGP', 'LEUCC', 
                   'ERYTH', 'HHB', 'HTE', 'VGM', 'TCMH', 'CCMH', 'NPLAQ', 'VPM', 'TCAM', 'TCAT', 'RTCAMT', 
                   'TQM', 'TQT', 'TP', 'FIB']
means = dict()
for col in to_mean:
    means[col] = round(df[col].mean(),2)
linea = df.loc[0].copy()

# -----------------------------------------------------------------------------------------
from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"
DOLIPRANE = "G:\Data\prevention_iatrogenie_centrale_2\doliprane.csv"

from import_tools import import_data
df = import_data(DOLIPRANE)
df.drop(['RTQMT','Unnamed: 0'], axis=1, inplace=True)

# ---
import importlib
import visualization
import machine_learning
importlib.reload(machine_learning); model_json = machine_learning.k_means(df,2)
importlib.reload(visualization); visualization.plot_cluster(df, model_json)

importlib.reload(machine_learning); model_json = machine_learning.hierarchical_clustering(df,'ward',20); importlib.reload(visualization); visualization.plot_cluster(df, model_json)

importlib.reload(machine_learning); model_json = machine_learning.meanshift(df,6); importlib.reload(visualization); visualization.plot_cluster(df, model_json)
 
importlib.reload(machine_learning); model_json = machine_learning.linear_regression(df); importlib.reload(visualization); visualization.plot_regression(df, model_json)
  
# --------------------------------------------------------------------------
from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')
from import_tools import import_data
PROCESSED_DATA = "G:\Data\prevention_iatrogenie_centrale_2\processed_data.csv"
df = import_data(PROCESSED_DATA)
from machine_learning import *

cluster_meds(df, ML_model=k_means, model_name='k_means')


from os import chdir
from gc import collect
chdir('G:\\Data\\prevention_iatrogenie_centrale_2\\Datos_github\\Fil Rouge\\Fil Rouge')
PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\grouped_data.csv"
from import_tools import import_random_subset, import_json
df = import_random_subset(PATH_DATA, size=500000)
dol = df.groupby(['CODE_ATC', 'VOIE_ADMIN', 'UNITE_PRESC']).get_group(('N02BE01', 'ORALE', 'mg'))
PATH_LR = r'G:\Data\prevention_iatrogenie_centrale_2\Models\linear_regression'
lr_params = import_json(PATH_LR + '\\'+ str(('N02BE01', 'ORALE', 'mg'))+'.json')
from warning_system import *
_, line = prepare_line(dol.iloc[12])
warning_lr(line, lr_params)




from simulation import test
test()



from import_tools import import_data
df = import_data(PROCESSED_DATA)
from machine_learning import k_means, cluster_meds
cluster_meds(df, ML_model=k_means, model_name='k_means')