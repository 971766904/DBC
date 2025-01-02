#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/25 16:13
# @Author  : zhongyu
# @Site    : 
# @File    : pca_tsne_distribution.py

'''
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np
from jddb.file_repo import FileRepo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def scaler_for_calculation(set_file_repo, shots):
    tag_list = set_file_repo.get_tag_list(shots[0])
    shots_data = pd.DataFrame(columns=tag_list)  # empty dataframe
    for shot in shots:
        data_raw = set_file_repo.read_data(shot, tag_list)
        shots_data = pd.concat([shots_data, pd.DataFrame(data_raw)], ignore_index=True)

    # Standardize the features for each dataset
    # shots_data['alarm_tag'] = shots_data['alarm_tag'].astype(int)
    X0 = shots_data.loc[shots_data['alarm_tag'] == 0].drop('alarm_tag', axis=1)
    X1 = shots_data.loc[shots_data['alarm_tag'] == 1].drop('alarm_tag', axis=1)
    scaler = StandardScaler()
    X0_scaled = scaler.fit_transform(X0)
    X1_scaled = scaler.fit_transform(X1)
    return X0_scaled, X1_scaled


def calculate_pca_0_1(X0_scaled1, X1_scaled1):
    n_components = 2  # You can adjust this
    pca1 = PCA(n_components=n_components)
    X1_pca = pca1.fit_transform(X1_scaled1)
    pca0 = PCA(n_components=n_components)
    X0_pca = pca0.fit_transform(X0_scaled1)
    return X0_pca, X1_pca


def calculate_tsne_0_1(X0_scaled1, X1_scaled1):
    tsne0 = TSNE(n_components=2, perplexity=30.0, n_iter=300, random_state=42)
    tsne1 = TSNE(n_components=2, perplexity=30.0, n_iter=300, random_state=42)
    X0_tsne = tsne0.fit_transform(X0_scaled1)
    X1_tsne = tsne1.fit_transform(X1_scaled1)
    return X0_tsne, X1_tsne


if __name__ == '__main__':
    # %%
    # load file
    train_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_1k_5k_label//$shot_2$00//')
    cate_1 = pd.read_csv('..//..//file_repo//info//std_info//category//cate_1.csv')
    cate_2 = pd.read_csv('..//..//file_repo//info//std_info//category//cate_2.csv')
    cate_3 = pd.read_csv('..//..//file_repo//info//std_info//category//cate_3.csv')
    cate_4 = pd.read_csv('..//..//file_repo//info//std_info//category//cate_4.csv')
    shots1 = cate_1['shot']
    shots2 = cate_2['shot']
    shots3 = cate_3['shot']
    shots4 = cate_4['shot']

    # %%
    #
    X0_scaled1, X1_scaled1 = scaler_for_calculation(train_file_repo, shots1)
    X0_scaled2, X1_scaled2 = scaler_for_calculation(train_file_repo, shots2)
    X0_scaled3, X1_scaled3 = scaler_for_calculation(train_file_repo, shots3)
    X0_scaled4, X1_scaled4 = scaler_for_calculation(train_file_repo, shots4)

    # %%
    # Apply PCA for each dataset
    X0_pca1, X1_pca1 = calculate_tsne_0_1(X0_scaled1, X1_scaled1)
    X0_pca2, X1_pca2 = calculate_tsne_0_1(X0_scaled2, X1_scaled2)
    X0_pca3, X1_pca3 = calculate_tsne_0_1(X0_scaled3, X1_scaled3)
    X0_pca4, X1_pca4 = calculate_tsne_0_1(X0_scaled4, X1_scaled4)

    # %%
    # Create a scatter plot
    point_s = 2
    alpha_a = 0.5
    n = 10000
    myfont = fm.FontProperties(family='Times New Roman', size=16, weight='bold')
    font = {'family': 'Times New Roman', 'size': 16, 'weight': 'black'}
    plt.rc('font', **font)
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X0_pca1[:, 0], y=X0_pca1[:, 1], s=point_s, label='cate 1')
    plt.scatter(x=X0_pca2[:, 0], y=X0_pca2[:, 1],  s=point_s, label='cate 2')
    plt.scatter(x=X0_pca3[:, 0], y=X0_pca3[:, 1],  s=point_s, label='cate 3')
    plt.scatter(x=X0_pca4[:, 0], y=X0_pca4[:, 1],  s=point_s, label='cate 4')
    plt.title("t_sne Visualization of Combined Undis Datasets")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X1_pca1[:, 0], y=X1_pca1[:, 1],  s=point_s, label='cate 1', alpha=alpha_a)
    plt.scatter(x=X1_pca2[:, 0], y=X1_pca2[:, 1],  s=point_s, label='cate 2', alpha=alpha_a)
    plt.scatter(x=X1_pca3[:, 0], y=X1_pca3[:, 1],  s=point_s, label='cate 3', alpha=alpha_a)
    plt.scatter(x=X1_pca4[:, 0], y=X1_pca4[:, 1],  s=point_s, label='cate 4', alpha=alpha_a)
    plt.title("t_sne Visualization of Combined dis Datasets")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    #%%
    # Use the ip info npy file to get the shots
    ip_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    ip_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    ip_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    ip_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    ip_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    #%%
    # get data from file_repo by shots
    X0_scaled1, X1_scaled1 = scaler_for_calculation(train_file_repo, ip_1)
    X0_scaled2, X1_scaled2 = scaler_for_calculation(train_file_repo, ip_2)
    X0_scaled3, X1_scaled3 = scaler_for_calculation(train_file_repo, ip_3)
    X0_scaled4, X1_scaled4 = scaler_for_calculation(train_file_repo, ip_4)
    X0_scaled5, X1_scaled5 = scaler_for_calculation(train_file_repo, ip_5)
    #%%
    # Apply tsne for each dataset and the whole dataset shots
    X0_tsne1, X1_tsne1 = calculate_tsne_0_1(X0_scaled1, X1_scaled1)
    X0_tsne2, X1_tsne2 = calculate_tsne_0_1(X0_scaled2, X1_scaled2)
    X0_tsne3, X1_tsne3 = calculate_tsne_0_1(X0_scaled3, X1_scaled3)
    X0_tsne4, X1_tsne4 = calculate_tsne_0_1(X0_scaled4, X1_scaled4)
    X0_tsne5, X1_tsne5 = calculate_tsne_0_1(X0_scaled5, X1_scaled5)



