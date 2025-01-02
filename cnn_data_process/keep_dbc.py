#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/16 16:46
# @Author  : zhongyu
# @Site    : 
# @File    : keep_dbc.py

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
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//training_shots_1.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//val_shots_1.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')
    print('training_shots:', len(training_shots))
    print('valid_shots:', len(valid_shots))
    print('training info:')
    original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    print('valid info:')
    original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)

    # # %%`
    # # plan B:
    # print('====================')
    # wrong_predict_shots_2 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots_train_1.npy')
    # wrong_predict_shots_1 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots_val_1.npy')
    #
    # # Exclude wrong_predict_shots_2 from training_shots_dbc before sampling
    # training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    # valid_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]
    # training_shots_dbc_excluded = training_shots_dbc[~training_shots_dbc['shot'].isin(wrong_predict_shots_2)]
    #
    # # Sort the undisrupted shots in training_shots_dbc in ascending order of dbc
    # undisrupted_shots = training_shots_dbc_excluded[training_shots_dbc_excluded['IsDisrupt'] == 0].sort_values('dbc')
    #
    # for i in range(1, 11):
    #     print('====================')
    #     # Remove i*10 undisrupted shots from training_shots and add them to valid_shots
    #     removed_shots = undisrupted_shots.head(i * 50)
    #     training_shots = np.setdiff1d(training_shots, removed_shots['shot'])
    #     valid_shots = np.concatenate((valid_shots, removed_shots['shot']))
    #
    #     # Calculate dbc of training_shots
    #     training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    #     new_dbc = training_shots_dbc['dbc'].sum()
    #
    #     # If dbc is less than 7532, add shots from valid_shots to training_shots
    #     while new_dbc < 69594:
    #         # Get dbc of valid_shots
    #         valid_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]
    #
    #         # Calculate the dbc difference
    #         dbc_diff = 69594 - new_dbc
    #
    #         # Get shots from valid_shots whose dbc is less than or equal to dbc_diff
    #         added_shots = valid_shots_dbc[valid_shots_dbc['dbc'] <= dbc_diff]
    #
    #         # If there are no such shots, break the loop
    #         if added_shots.empty:
    #             break
    #
    #         # Add the shot with the highest dbc to training_shots
    #         added_shot = added_shots.loc[added_shots['dbc'].idxmax()]
    #         training_shots = np.concatenate((training_shots, [added_shot['shot']]))
    #         valid_shots = np.setdiff1d(valid_shots, [added_shot['shot']])
    #
    #         # Recalculate the dbc of training_shots
    #         training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    #         new_dbc = training_shots_dbc['dbc'].sum()
    #     # training_shots = np.concatenate((training_shots, wrong_predict_shots_1))
    #     print('train shots:')
    #     original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    #     print('valid shots:')
    #     original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
    #     if len(training_shots) != len(set(training_shots)):
    #         print('-----repeated shots in training_shots')
    #     # check if valid_shots have the repeated shots
    #     if len(valid_shots) != len(set(valid_shots)):
    #         print('-----repeated shots in valid_shots')
    #
    #     # # save the new training shots and valid shots
    #     # np.save('..//..//file_repo//info//dataset//shots_ratio/training_shots_{}.npy'.format(i), training_shots)
    #     # np.save('..//..//file_repo//info//dataset//shots_ratio/valid_shots_{}.npy'.format(i), valid_shots)


    # %%
    # display the information of each new training shots and valid shots
    # new a df to store the original dbc, dis_ratio, num_train of each new training shots
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(10):
        print('====================')
        print('Iteration:', i + 1)
        training_shots = np.load('..//..//file_repo//info//dataset//shots_ratio/training_shots_{}.npy'.format(i+1))
        valid_shots = np.load('..//..//file_repo//info//dataset//shots_ratio/valid_shots_{}.npy'.format(i+1))
        print('train shots:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid shots:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
    print(df)
    # save the dataframe
    df.to_csv('..//..//file_repo//info//dataset//shots_ratio//power_shots_ratio_info.csv')
