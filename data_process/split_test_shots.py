#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/23 17:29
# @Author  : zhongyu
# @Site    : 
# @File    : split_test_shots.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import pandas as pd
from jddb.file_repo import FileRepo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def dataset_shots_split(shots_1,shots_info):
    shots_is_disrupt = shots_info[shots_info['shot'].isin(shots_1)]['IsDisrupt']
    train_shots, test_shots, _, _ = \
        train_test_split(shots_1, shots_is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=shots_is_disrupt)
    return train_shots, test_shots



if __name__ == '__main__':
    # %%
    # load file
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')

    # %%
    # load ip info files
    shots_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    shots_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    shots_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    shots_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')

    # %%
    # calculate the shots disruption rate of 5 shots individually
    shots_1_dis = dbc_shots_info[dbc_shots_info['shot'].isin(shots_1)]['IsDisrupt'].mean()
    shots_2_dis = dbc_shots_info[dbc_shots_info['shot'].isin(shots_2)]['IsDisrupt'].mean()
    shots_3_dis = dbc_shots_info[dbc_shots_info['shot'].isin(shots_3)]['IsDisrupt'].mean()
    shots_4_dis = dbc_shots_info[dbc_shots_info['shot'].isin(shots_4)]['IsDisrupt'].mean()
    shots_5_dis = dbc_shots_info[dbc_shots_info['shot'].isin(shots_5)]['IsDisrupt'].mean()

    #%%
    # split each shots in to train and test with sklearn split,according to IsDisrupt
    shots_is_disrupt = dbc_shots_info[dbc_shots_info['shot'].isin(shots_1)]['IsDisrupt']
    train_shots, test_shots, _, _ = \
        train_test_split(shots_1, shots_is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=shots_is_disrupt)
    train_1, test_1 = dataset_shots_split(shots_1,dbc_shots_info)
    train_2, test_2 = dataset_shots_split(shots_2,dbc_shots_info)
    train_3, test_3 = dataset_shots_split(shots_3,dbc_shots_info)
    train_4, test_4 = dataset_shots_split(shots_4,dbc_shots_info)

    #%%
    # save test_1-4
    np.save('..//..//file_repo//info//split_dataset_info//test_1.npy',test_1)
    np.save('..//..//file_repo//info//split_dataset_info//test_2.npy',test_2)
    np.save('..//..//file_repo//info//split_dataset_info//test_3.npy',test_3)
    np.save('..//..//file_repo//info//split_dataset_info//test_4.npy',test_4)

    #%%
    # concatenate train_1-4, concatenate test_1-4 with shots_5
    train_shots_1 = np.concatenate((train_1,train_2,train_3,train_4))
    test_shots_1 = np.concatenate((test_1,test_2,test_3,test_4,shots_5))
    # save train shots and test shots 1
    np.save('..//..//file_repo//info//split_dataset_info//train_shots_1.npy',train_shots_1)
    np.save('..//..//file_repo//info//split_dataset_info//test_shots_1.npy',test_shots_1)



