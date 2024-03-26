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
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X0_pca1[:, 0], y=X0_pca1[:, 1], s=10, label='cate 1')
    plt.scatter(x=X0_pca2[:, 0], y=X0_pca2[:, 1],  s=10, label='cate 2')
    plt.scatter(x=X0_pca3[:, 0], y=X0_pca3[:, 1],  s=10, label='cate 3')
    plt.scatter(x=X0_pca4[:, 0], y=X0_pca4[:, 1],  s=10, label='cate 4')
    plt.title("PCA Visualization of Combined Undis Datasets")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X1_pca1[:, 0], y=X1_pca1[:, 1],  s=10, label='cate 1')
    plt.scatter(x=X1_pca2[:, 0], y=X1_pca2[:, 1],  s=10, label='cate 2')
    plt.scatter(x=X1_pca3[:, 0], y=X1_pca3[:, 1],  s=10, label='cate 3')
    plt.scatter(x=X1_pca4[:, 0], y=X1_pca4[:, 1],  s=10, label='cate 4')
    plt.title("PCA Visualization of Combined dis Datasets")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()
