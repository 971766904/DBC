#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/21 21:16
# @Author  : zhongyu
# @Site    : 
# @File    : add_test_back.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import copy
import os
from shots_selection.select_strategy import get_dataset_info

if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//dbc_down//training_shots_2.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//dbc_down//valid_shots_2.npy')
    test_back_shots = np.load('..//..//file_repo//info//dataset//test_new_back//test_back_shots.npy')

    #%%
    # add test back shots to training shots
    test_back_info = dbc_shots_info[dbc_shots_info['shot'].isin(test_back_shots)]
    test_back_dbc = test_back_info.sort_values(by='dbc')
    for i in range(10):
        print('====================')
        print('Iteration:', i + 1)
        # Select shots from test_back_shots
        add_test_back_shots = test_back_dbc.iloc[:10 * (i + 1)]['shot']
        # Combine test_back_shots and training_shots
        new_training_shots = np.concatenate((training_shots, add_test_back_shots))

        # show info
        print('training info:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, new_training_shots)
        print('valid info:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
        # check if training_shots have the repeated shots
        if len(new_training_shots) != len(set(new_training_shots)):
            print('-----repeated shots in training_shots')
        # check if valid_shots have the repeated shots
        if len(valid_shots) != len(set(valid_shots)):
            print('-----repeated shots in valid_shots')
        # # Save the new training shots and valid shots
        # np.save('..//..//file_repo//info//dataset//add_test_back//training_shots_{}.npy'.format(i+1), new_training_shots)
        # np.save('..//..//file_repo//info//dataset//add_test_back//valid_shots_{}.npy'.format(i+1), valid_shots)

    # %%
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(10):
        print('====================')
        training_shots = np.load('..//..//file_repo//info//dataset//add_test_back//training_shots_{}.npy'.format(i + 1))
        valid_shots = np.load('..//..//file_repo//info//dataset//add_test_back//valid_shots_{}.npy'.format(i + 1))
        print('training info:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid info:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
        # creat new folder named _temp_test_{}.formate(i+1)
        os.makedirs('..//..//file_repo//info//dataset//add_test_back//_temp_test_{}'.format(i + 1))

    print(df)
    # df.to_csv('..//..//file_repo//info//dataset//add_test_back//add_test_info.csv')


