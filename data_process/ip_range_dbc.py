#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/4 0:19
# @Author  : zhongyu
# @Site    : 
# @File    : ip_range_dbc.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import pandas as pd

if __name__ == '__main__':
    # %%
    # load the old dbc info csv
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')

    # load ip range npy file
    ip_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    ip_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    ip_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    ip_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    ip_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    # %%
    # get shots info of 5 shots range
    ip_1_info = dbc_shots_info[dbc_shots_info['shot'].isin(ip_1)]
    ip_2_info = dbc_shots_info[dbc_shots_info['shot'].isin(ip_2)]
    ip_3_info = dbc_shots_info[dbc_shots_info['shot'].isin(ip_3)]
    ip_4_info = dbc_shots_info[dbc_shots_info['shot'].isin(ip_4)]
    ip_5_info = dbc_shots_info[dbc_shots_info['shot'].isin(ip_5)]

    # %%
    # calculate the mean and std of bt and p of 5 shots range,new a column named new_dbc
    # new_db=bt-bt_mean/std(bt) + p-p_mean/std(p)
    bt_mean = ip_1_info['bt'].min()
    bt_std = ip_1_info['bt'].std()
    p_mean = ip_1_info['p'].min()
    p_std = ip_1_info['p'].std()
    ip_1_info['new_dbc'] = (ip_1_info['bt'] - bt_mean) / bt_std + (ip_1_info['p'] - p_mean) / p_std

    # %%
    # put the new_dbc to dbc_shots_info and save as csv file
    dbc_shots_info['new_dbc'] = 0
    dbc_shots_info.loc[dbc_shots_info['shot'].isin(ip_1), 'new_dbc'] = ip_1_info['new_dbc']

