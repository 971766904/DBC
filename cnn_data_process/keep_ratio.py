#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/15 10:57
# @Author  : zhongyu
# @Site    : 
# @File    : keep_ratio.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import os
from shots_change import get_dataset_info

if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix'
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//dbc_shots//training_shots_1.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//dbc_shots//valid_shots_1.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')
    print('training_shots:', len(training_shots))
    print('valid_shots:', len(valid_shots))

    # # %%
    # # calculate the ratio of disruption shots in training shots and valid shots
    # training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    # valid_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]
    # test_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(test_shots)]
    # training_dis_ratio = training_shots_dbc['IsDisrupt'].sum() / len(training_shots)
    # valid_dis_ratio = valid_shots_dbc['IsDisrupt'].sum() / len(valid_shots)
    # test_dis_ratio = test_shots_dbc['IsDisrupt'].sum() / len(test_shots)
    # print('training_dis_ratio:', training_dis_ratio)
    # print('valid_dis_ratio:', valid_dis_ratio)
    # print('test_dis_ratio:', test_dis_ratio)
    #
    # #%%
    # # Calculate the number of disruption and non-disruption shots needed in each 50-shot subset
    # num_disruption_shots = int(50 * test_dis_ratio)
    # num_non_disruption_shots = 50 - num_disruption_shots
    #
    # # Initialize a list to store the subsets
    # training_subsets = []
    #
    # for i in range(9):
    #     print('====================')
    #     print('Iteration:', i + 1)
    #     # Select disruption shots from training_shots
    #     disruption_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 1].sample(num_disruption_shots)
    #     # Select non-disruption shots from training_shots
    #     non_disruption_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 0].sample(num_non_disruption_shots)
    #     # Combine disruption and non-disruption shots to form a subset
    #     subset = np.concatenate((disruption_shots['shot'], non_disruption_shots['shot']))
    #     # Add the subset to the list of subsets
    #     training_subsets.append(subset)
    #     # Remove the selected shots from training_shots
    #     training_shots = np.setdiff1d(training_shots, subset)
    #     # move the subset to valid_shots
    #     valid_shots = np.concatenate((valid_shots, subset))
    #     print('training info:')
    #     original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    #     print('valid info:')
    #     original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
    #     # Save the new training shots and valid shots
    #     # np.save('..//..//file_repo//info//dataset//dbc_shots//training_shots_{}.npy'.format(i+2), training_shots)
    #     # np.save('..//..//file_repo//info//dataset//dbc_shots//valid_shots_{}.npy'.format(i+2), valid_shots)
    #
    # # Now, training_subsets contains 9 subsets of 50 shots each, and training_shots has been reduced accordingly

    #%%
    # display the information of each new training shots and valid shots
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(10):
        print('====================')
        print('Iteration:', i + 1)
        training_shots = np.load('..//..//file_repo//info//dataset//dbc_shots//training_shots_{}.npy'.format(i+1))
        valid_shots = np.load('..//..//file_repo//info//dataset//dbc_shots//valid_shots_{}.npy'.format(i+1))
        print('train shots:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid shots:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)

    print(df)
    # save the dataframe
    df.to_csv('..//..//file_repo//info//dataset//dbc_shots//dbc_shots_info.csv')

    # by wrong predict
    print('====================')
    bw_training_shots = np.load('..//..//file_repo//info//dataset//training_shots.npy')
    bw_valid_shots = np.load('..//..//file_repo//info//dataset//valid_shots.npy')
    print('train shots:')
    original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, bw_training_shots)
    print('valid shots:')
    original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, bw_valid_shots)

