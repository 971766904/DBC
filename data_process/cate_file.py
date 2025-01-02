#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/20 11:07
# @Author  : zhongyu
# @Site    : 
# @File    : cate_file.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import pandas as pd
from jddb.file_repo import FileRepo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    # %%
    # load file
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')

    # %%
    # show the whole shots ip and bt distribution with scatter
    dis_shots = dbc_shots_info[dbc_shots_info['IsDisrupt'] == 1]
    undis_shots = dbc_shots_info[dbc_shots_info['IsDisrupt'] == 0]
    plt.figure(figsize=(8, 6))
    # plt.scatter(dbc_shots_info['ip'], dbc_shots_info['bt'], color='blue', alpha=0.7)
    plt.scatter(dis_shots['ip'], dis_shots['bt'], color='red', alpha=0.7)
    plt.scatter(undis_shots['ip'], undis_shots['bt'], color='green', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('bt')
    plt.title('ip and bt distribution of all shots')
    plt.show()

    # %%
    # show the whole shots ip,bt and power distribution with scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(dbc_shots_info['ip'], dbc_shots_info['bt'], c=dbc_shots_info['p'], cmap='coolwarm', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('bt')
    plt.title('ip and bt distribution of all shots')
    plt.colorbar()
    plt.show()

    #%%
    # generate 5 category shots according to ip, bt and power by kmeans
    from sklearn.cluster import KMeans
    dis_shots_info = dbc_shots_info[dbc_shots_info['IsDisrupt'] == 1]
    undis_shots_info = dbc_shots_info[dbc_shots_info['IsDisrupt'] == 0]
    dis_shots = dis_shots_info[['ip', 'bt', 'p']]
    undis_shots = undis_shots_info[['ip', 'bt', 'p']]
    shots = undis_shots[['ip', 'bt', 'p']]
    shots_info = undis_shots_info
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(shots)
    shots_info['category'] = kmeans.labels_
    # show the category distribution
    plt.figure(figsize=(8, 6))
    plt.scatter(shots_info['ip'], shots_info['bt'], c=shots_info['category'], cmap='coolwarm', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('bt')
    plt.title('ip and bt distribution of all shots')
    plt.colorbar()
    plt.show()
