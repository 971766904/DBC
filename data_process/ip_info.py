#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/19 21:44
# @Author  : zhongyu
# @Site    : 
# @File    : ip_info.py

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

    # show the whole shots ip distribution with histogram
    plt.figure(figsize=(8, 6))
    plt.hist(dbc_shots_info['ip'], bins=100, color='blue', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('count')
    plt.title('ip distribution of all shots')


    # show the whole shots with IsDisrupt is FALSE undistruption shots is distribution with histogram
    plt.hist(dbc_shots_info[dbc_shots_info['IsDisrupt'] == 0]['ip'], bins=100, color='green', alpha=0.7)


    # show the whole shots with IsDisrupt is TRUE disruption shots ip distribution with histogram
    plt.hist(dbc_shots_info[dbc_shots_info['IsDisrupt'] == 1]['ip'], bins=100, color='red', alpha=0.7)
    plt.show()

    # show the legend
    plt.legend(['all', 'undisruption', 'disruption'])

    # ip select: 110, 130, 147, 165, 186. cut all shots into 5 parts
    ip_1 = dbc_shots_info[dbc_shots_info['ip'] < 130]
    ip_2 = dbc_shots_info[(dbc_shots_info['ip'] >= 130) & (dbc_shots_info['ip'] < 147)]
    ip_3 = dbc_shots_info[(dbc_shots_info['ip'] >= 147) & (dbc_shots_info['ip'] < 165)]
    ip_4 = dbc_shots_info[(dbc_shots_info['ip'] >= 165) & (dbc_shots_info['ip'] < 186)]
    ip_5 = dbc_shots_info[dbc_shots_info['ip'] >= 186]
    # show 130, 147, 165, 186 in the figure with a red line
    plt.axvline(130, color='red', linestyle='--')
    plt.axvline(147, color='red', linestyle='--')
    plt.axvline(165, color='red', linestyle='--')
    plt.axvline(186, color='red', linestyle='--')
    plt.show()

    # save the ip_1, ip_2, ip_3, ip_4, ip_5 as npy file
    np.save('..//..//file_repo//info//ip_info//ip_1.npy', ip_1['shot'])
    np.save('..//..//file_repo//info//ip_info//ip_2.npy', ip_2['shot'])
    np.save('..//..//file_repo//info//ip_info//ip_3.npy', ip_3['shot'])
    np.save('..//..//file_repo//info//ip_info//ip_4.npy', ip_4['shot'])
    np.save('..//..//file_repo//info//ip_info//ip_5.npy', ip_5['shot'])

    # %%
    # show the shots in test shots wrong predicted
    # load the npy file which contain shots number
    shot_result = np.load('..//..//file_repo//info//ip_info//test_shots.npy')
    # shot_result = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    # get these shots info in dbc_shots_info
    test_shots_info = dbc_shots_info[dbc_shots_info['shot'].isin(shot_result)]
    plt.figure(figsize=(8, 6))
    plt.hist(test_shots_info['bt'], bins=100, color='blue', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('count')
    plt.title('ip distribution of test set wrong shots')





