#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/3 21:49
# @Author  : zhongyu
# @Site    : 
# @File    : choose_shot_from_test.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import pandas as pd

if __name__ == '__main__':
    # %%
    # load the dbc info csv
    # qa = a^2*bt/2.1/ip
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')
    # new a column named qa, and calculate the value by bt*ip
    dbc_shots_info['qa'] = dbc_shots_info['bt'] * dbc_shots_info['ip']

    # %%
    # load the shots npy file
    wrong_shots = np.load('..//..//file_repo//info//ip_info//test_shots.npy')
    test_shots = np.load('..//..//file_repo//info//ip_info//ip_5.npy')

    # %%
    # plot the histgram figure
    test_shots_info = dbc_shots_info[dbc_shots_info['shot'].isin(test_shots)]
    plt.figure(figsize=(8, 6))
    plt.hist(test_shots_info['qa'], bins=100, color='blue', alpha=0.7)
    plt.xlabel('ip')
    plt.ylabel('count')
    plt.title('ip distribution of test set wrong shots')

    # %%
    # choose shots according to the ip distribution
    # only 50% of wrong_shots whose disrupiton is 0 are chosen
    wrong_shots_info = dbc_shots_info[dbc_shots_info['shot'].isin(wrong_shots)]
    wrong_shots_undisrupt = wrong_shots_info[wrong_shots_info['IsDisrupt'] == 0]
    wrong_shots_undisrupt = wrong_shots_undisrupt.sample(frac=0.5)
    # save the chosen shots no as npy file
    np.save('..//..//file_repo//info//ip_info//chosen_shots_undis.npy', wrong_shots_undisrupt['shot'])


