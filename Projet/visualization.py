'''
------------------------------------------------------------------------------------------------
                                       Visualization Module
------------------------------------------------------------------------------------------------

This module contains functions to create and analyse clusters of meds

'''
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, Any

from sklearn.cluster import KMeans



def example_k_means(df: DataFrame, n_clusters=7) -> dict():
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df)
    return kmeans.labels_.astype(int), kmeans.cluster_centers_

def example_mobius():
    theta = np.linspace(0, 2*np.pi,90)
    w = np.linspace(-0.25,0.25,5)
    w, theta = np.meshgrid(w, theta)
    phi = 0.5*theta
    r = 1 + w*np.cos(phi)
    x = np.ravel(r*np.cos(theta))
    y = np.ravel(r*np.sin(theta))
    z = np.ravel(w*np.sin(phi))
    df = pd.DataFrame(np.array([x,y,z]).transpose(), columns=['x','y','z'])
    return df

def compress_centers(df, tsne_3d, tsne_2d):
    x = np.array(df)
    plot_df = pd.DataFrame(TSNE(n_components=3, learning_rate='auto',init='random').fit_transform(x), columns = ['3x','3y','3z'])
    plot_df = plot_df.join(pd.DataFrame(TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(x), columns=['2x','2y']))
    return plot_df

#def compress_centers(df, centers):
#    x = np.array(df)
#    np.append(x,centers)
#    plot_df = pd.DataFrame(TSNE(n_components=3, learning_rate='auto',init='random').fit_transform(x), columns = ['3x','3y','3z'])
#    plot_df = plot_df.join(pd.DataFrame(TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(x), columns=['2x','2y']))
#    plot_centers = plot_df.tail(len(centers))
#    plot_df = plot_df.drop(plot_df.tail(len(centers)).index, inplace=True)
#    return plot_df, plot_centers


def visualize() -> None:
    df = example_mobius()
    
    plot_df, plot_centers = compress_centers(df,centers)

    fig = plt.figure(figsize=[15,15])
    for i in range(3):
        labels, centers = example_k_means(df, 3+i)

        #ax3D = fig.add_subplot(3,2,1+2*i, projection='3d')
        #ax3D.scatter3D(plot_df['3x'],plot_df['3y'],plot_df['3z'], c=labels)
        #ax2D = fig.add_subplot(3,2,2+2*i)

        #ax2D.scatter(plot_centers['2x'],plot_centers['2y'], marker='^')
        #ax2D.scatter(plot_df['2x'],plot_df['2y'], c=labels)


    plt.savefig("output.png")
    print("Finished plotting. Muahaha")
    

def plot_regression(data: DataFrame, model: dict()) -> None:
    qte = data['QTE_PRESC']
    data_copy = data.copy()
    X = data_copy.drop(['QTE_PRESC'], axis = 1)
    point0 = [0,model['intercept']]
    PCA_3D = PCA(n_components=3)
    PCA_2D = PCA(n_components=2)
    data_t = PCA_2D.fit_transform(X)
    data_t_3d = PCA_3D.fit_transform(X)
    fig = plt.figure(figsize=[16,6])
    coef_data = model['coef']
    point1 = PCA_2D.transform(coef_data.reshape(1,-1))
    fig.suptitle('Visualization de la Regression Lineaire')

    ax_quant_3D = fig.add_subplot(1,2,1, projection='3d')
    ax_quant_2D = fig.add_subplot(1,2,2)

    cmap = plt.get_cmap('viridis')
    #norm = plt.Normalize(vmin=0, vmax=center_2d.shape[0]-1)

    ax_quant_3D.view_init(-140, 60)
    plot3 = ax_quant_2D.scatter(data_t[:, 0], data_t[:, 1])
    print(point0,point1)
    ax_quant_2D.plot(point0,*point1)

    ax_quant_3D.set_title('Linear 3D')
    ax_quant_3D.scatter3D(data_t_3d[:, 0],data_t_3d[:, 1],data_t_3d[:, 2])
    ax_quant_2D.set_title('Linear 2D')

    plt.savefig("output_regression.png")
    
    
def plot_cluster(data: DataFrame, model: dict()) -> None:
    n = len(model['centers'])
    qte = data['QTE_PRESC']
    data_copy = data.copy()
    X = data_copy.drop(['QTE_PRESC'], axis = 1)
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    PCA_3D = PCA(n_components=3)
    PCA_2D = PCA(n_components=2)
    data_t = PCA_2D.fit_transform(X)
    data_t_3d = PCA_3D.fit_transform(X)
    fig = plt.figure(figsize=[16,6])
    fig.suptitle('Visualization des clusters par reduction de dimensions')

    ax_quant_3D = fig.add_subplot(1,2,1, projection='3d')
    ax_quant_2D = fig.add_subplot(1,2,2)

    center_data = np.delete(model['centers'],data_copy.columns.get_loc('QTE_PRESC'),axis=1)
    center_2d = PCA_2D.transform(center_data)
    center_3d = PCA_3D.transform(center_data)

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=0, vmax=center_2d.shape[0]-1)

    ax_quant_3D.view_init(-140, 60)
    plot3 = ax_quant_2D.scatter(data_t[:, 0], data_t[:, 1], c=model['labels'])
    for point, label in zip(data_t_3d,model['labels']):
        ax_quant_2D.plot([point[0],center_2d[label][0]],[point[1],center_2d[label][1]], alpha=0.01, c=cmap(norm(label)))
    
    ax_quant_3D.set_title('Cluster 3D')
    ax_quant_3D.scatter3D(data_t_3d[:, 0],data_t_3d[:, 1],data_t_3d[:, 2], c=model['labels'])
    for point, label in zip(data_t_3d,model['labels']):
        ax_quant_3D.plot([point[0],center_3d[label][0]],[point[1],center_3d[label][1]],zs=[point[2],center_3d[label][2]], alpha=0.01, c=cmap(norm(label)))
    plt.legend(handles=plot3.legend_elements()[0], labels=['Class '+str(k+1) for k in range(n)])
    ax_quant_2D.set_title('Clusters 2D')

    plt.savefig("output_cluster.png")
    #plt.show()    