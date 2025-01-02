#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/15 14:14
# @Author  : zhongyu
# @Site    : 
# @File    : keep_shots.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import copy
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

    #  # %%
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
    # print('training info:')
    # original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    # print('valid info:')
    # original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
    #
    # # %%
    # # Initialize a list to store the new training shots for each dis_ratio
    # new_training_shots_list = []
    # i = 0
    #
    # # For each new dis_ratio from 0.1 to 0.2
    # for new_dis_ratio in np.arange(0.1, 0.21, 0.01):
    #     print('====================')
    #     print('ratio:', new_dis_ratio)
    #     print('Iteration:', i)
    #
    #     temp_training_shots = copy.deepcopy(training_shots)
    #     temp_valid_shots = copy.deepcopy(valid_shots)
    #     # Calculate the current number of disruption shots in training_shots
    #     current_disruption_shots = int(len(temp_training_shots) * test_dis_ratio)
    #
    #     # Calculate the required number of disruption shots in training_shots
    #     required_disruption_shots = int(1080 * new_dis_ratio)
    #
    #     # Calculate the difference
    #     difference = required_disruption_shots - current_disruption_shots
    #
    #     # Adjust the number of disruption shots
    #     if difference > 0:
    #         # Add disruption shots from valid_shots to training_shots
    #         added_shots = valid_shots_dbc[valid_shots_dbc['IsDisrupt'] == 1].sample(difference)
    #         temp_training_shots = np.concatenate((temp_training_shots, added_shots['shot']))
    #         temp_valid_shots = np.setdiff1d(temp_valid_shots, added_shots['shot'])
    #         # Remove non-disruption shots from training_shots and add them to valid_shots
    #         removed_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 0].sample(difference)
    #         temp_training_shots = np.setdiff1d(temp_training_shots, removed_shots['shot'])
    #         temp_valid_shots = np.concatenate((temp_valid_shots, removed_shots['shot']))
    #     else:
    #         # Remove disruption shots from training_shots and add them to valid_shots
    #         removed_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 1].sample(-difference)
    #         temp_training_shots = np.setdiff1d(temp_training_shots, removed_shots['shot'])
    #         temp_valid_shots = np.concatenate((temp_valid_shots, removed_shots['shot']))
    #         # Add non-disruption shots from valid_shots to training_shots
    #         added_shots = valid_shots_dbc[valid_shots_dbc['IsDisrupt'] == 0].sample(-difference)
    #         temp_training_shots = np.concatenate((temp_training_shots, added_shots['shot']))
    #         temp_valid_shots = np.setdiff1d(temp_valid_shots, added_shots['shot'])
    #
    #     # Add the new training_shots to the list
    #     new_training_shots_list.append(temp_training_shots)
    #     print('train shots:')
    #     original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, temp_training_shots)
    #     print('valid shots:')
    #     original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, temp_valid_shots)
    #     # check if temp_training_shots have the repeated shots
    #     if len(temp_training_shots) != len(set(temp_training_shots)):
    #         print('-----repeated shots in temp_training_shots')
    #     # check if temp_valid_shots have the repeated shots
    #     if len(temp_valid_shots) != len(set(temp_valid_shots)):
    #         print('-----repeated shots in temp_valid_shots')
    #
    #     # # save the new training shots and valid shots with different file name
    #     # np.save('..//..//file_repo//info//dataset//dbc_ratio//dbc_training_shots_{}.npy'.format(i+1), temp_training_shots)
    #     # np.save('..//..//file_repo//info//dataset//dbc_ratio//dbc_valid_shots_{}.npy'.format(i+1), temp_valid_shots)
    #     i += 1

    #%%
    # display the information of each new training shots and valid shots
    # new a df to store the original dbc, dis_ratio, num_train of each new training shots
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(10):
        print('====================')
        print('Iteration:', i + 1)
        training_shots = np.load('..//..//file_repo//info//dataset//dbc_ratio//dbc_training_shots_{}.npy'.format(i+1))
        valid_shots = np.load('..//..//file_repo//info//dataset//dbc_ratio//dbc_valid_shots_{}.npy'.format(i+1))
        print('train shots:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid shots:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)

    print(df)
    df.to_csv('..//..//file_repo//info//dbc_ratio_train_info.csv', index=False)
