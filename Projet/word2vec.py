
'''
------------------------------------------------------------------------------------------------
                                       Word2Vec Module
------------------------------------------------------------------------------------------------

This module is useful to establish the compatibility of different drugs.

'''

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from import_tools import import_splitted
from splitted_df_tools import mergeDF

from sklearn.decomposition import PCA

from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

PATH_DATA = "G:\Data\prevention_iatrogenie_centrale_2\securimed_filtered.csv"

def write_CODE_ATC(df : pd.DataFrame) -> None:
    # Writes a text, each sentence is the drugs of a same prescription
    group = df.groupby(['P_ORD_C_NUM'])
    file = open("CODE_ATC.txt","w+")
    for a in group:
        line = " ".join(a[1]['CODE_ATC'].to_list())
        file.write(line+". ")
    file.close()


def write_CODE_ATC_complete(df: pd.DataFrame) -> None:
    # Writes a text, each sentence is the drugs of a same prescription
    #df_list = import_splitted(PATH_DATA, n_samples=4)
    #df = mergeDF(df_list)
    #for df in df_list():
    df.dropna(subset=['CODE_ATC'], inplace=True)
    group = df.groupby(['P_ORD_C_NUM'])
    file = open("CODE_ATC_complete.txt","w")
    for a in group:
        line = " ".join(a[1]['CODE_ATC'].to_list())
        file.write(line+". ")
    file.close()


def word2vecModel(file: str) -> None:
    # Code from Sumedh Kadam, "Python, word embedding using Word2Vec", GeeksforGeeks
    # Reads the file
    text = open(file)
    t = text.read()
    data = []
    # iterate through each sentence in the file
    for i in sent_tokenize(t):
        temp = []
        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())
        data.append(temp)
    #Create Skip Gram model
    model = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100, window = 3, sg=1)
    model.save("word2vecComplete.model")




def display_pca_scatterplot(model, words=None, sample=0):
    # Code from web.standford.edu on the gensim word vector visualization
    if words == None:
        if sample>0:
            words = np.random.choice(list(model.wv.index_to_key), sample)
        else:
            words = [word for word in list(model.wv.index_to_key)]
    print(words)
    word_vectors = np.array([model.wv[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:,:2]
    plt.figure(figsize=(20,20))
    plt.scatter(twodim[:,0],twodim[:,1],edgecolors='k',c='r')
    #for word, (x,y) in zip(words,twodim):
    #   plt.text(x+0.001, y+0.001, word)
    plt.savefig("figure.png")



#write_CODE_ATC_complete(PATH_DATA)








