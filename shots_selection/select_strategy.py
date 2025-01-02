#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/20 10:42
# @Author  : zhongyu
# @Site    : 
# @File    : select_strategy.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import copy
import os


def get_dataset_info(dbc_info, dataset_shots):
    # orignal total dbc of training shots
    dataset_shots_dbc = dbc_info[dbc_info['shot'].isin(dataset_shots)]
    original_dbc = dataset_shots_dbc['dbc'].sum()
    print('original_dbc:', original_dbc)
    # calculate the ratio of disruption shots in training shots
    dataset_dis_ratio = dataset_shots_dbc['IsDisrupt'].sum() / len(dataset_shots)
    print('data_dis_ratio:', dataset_dis_ratio)
    # calculate length of training shots
    num_train = len(dataset_shots)
    print('num_dataset_shots:', num_train)
    return original_dbc, dataset_dis_ratio, num_train


def adjust_dbc(original_dbc, temp_training_shots, temp_valid_shots, dbc_info):
    temp_train_dbc = dbc_info[dbc_info['shot'].isin(temp_training_shots)]
    temp_valid_dbc = dbc_info[dbc_info['shot'].isin(temp_valid_shots)]
    if original_dbc < 8735:
        # Add shots
        added_shots = temp_valid_dbc[temp_valid_dbc['IsDisrupt'] == 1].sample(1)
        temp_training_shots = np.concatenate((temp_training_shots, added_shots['shot']))
        temp_valid_shots = np.setdiff1d(temp_valid_shots, added_shots['shot'])
    else:
        # Remove shots
        removed_shots = temp_train_dbc[temp_train_dbc['IsDisrupt'] == 0].sample(1)
        temp_training_shots = np.setdiff1d(temp_training_shots, removed_shots['shot'])
        temp_valid_shots = np.concatenate((temp_valid_shots, removed_shots['shot']))

    return temp_training_shots, temp_valid_shots


if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix'
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')
    power_dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//training_shots_1.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//val_shots_1.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')
    training_1 = np.load('..//..//file_repo//info//dataset//dbc_down//training_shots_2.npy')
    training_2 = np.load('..//..//file_repo//info//dataset//training_shots_1.npy')
    training_3 = np.load('..//..//file_repo//info//dataset//training_shots_2.npy')
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
    # # new 10 training shots and valid shots by
    # # concatenate training shots and valid shots, take 827 shots from the concatenated shots as training shots,
    # # and the rest as valid shots
    # temp_training_shots = np.concatenate((training_shots, valid_shots))
    # # # save temp_training_shots
    # # np.save('..//..//file_repo//info//dataset//train+val//training_val_shots.npy', temp_training_shots)
    # temp_training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(temp_training_shots)]
    # dis_shots = temp_training_shots_dbc[temp_training_shots_dbc['IsDisrupt'] == 1]
    # undis_shots = temp_training_shots_dbc[temp_training_shots_dbc['IsDisrupt'] == 0]
    #
    # #
    # ratio_dis = 0.21
    # i=1
    # for shots in range(100, 1300, 200):
    #     print('====================')
    #     print('Iteration:', shots)
    #     # choose shots*ratio_dis disruption shots and shots*(1-ratio_dis) undisruption shots
    #     dis_shots_sample = dis_shots.sample(int(shots * ratio_dis))
    #     undis_shots_sample = undis_shots.sample(int(shots * (1 - ratio_dis)))
    #     new_training_shots = np.concatenate((dis_shots_sample['shot'], undis_shots_sample['shot']))
    #     new_valid_shots = np.setdiff1d(temp_training_shots, new_training_shots)
    #
    #     # show info
    #     print('training info:')
    #     original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, new_training_shots)
    #     print('valid info:')
    #     original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, new_valid_shots)
    #     # check if training_shots have the repeated shots
    #     if len(new_training_shots) != len(set(new_training_shots)):
    #         print('-----repeated shots in training_shots')
    #     # check if valid_shots have the repeated shots
    #     if len(new_valid_shots) != len(set(new_valid_shots)):
    #         print('-----repeated shots in valid_shots')
    #     # # Save the new training shots and valid shots
    #     # np.save('..//..//file_repo//info//dataset//dbc_shots_200pre//training_shots_{}.npy'.format(i), new_training_shots)
    #     # np.save('..//..//file_repo//info//dataset//dbc_shots_200pre//valid_shots_{}.npy'.format(i), new_valid_shots)
    #     # i+= 1
    #%%
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(6):
        print('====================')
        training_shots = np.load('..//..//file_repo//info//dataset//dbc_shots_200pre//training_shots_{}.npy'.format(i + 1))
        valid_shots = np.load('..//..//file_repo//info//dataset//dbc_shots_200pre//valid_shots_{}.npy'.format(i + 1))
        print('training info:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(power_dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid info:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)

    print(df)
    df.to_csv('..//..//file_repo//info//dataset//dbc_shots_200pre//power_200pre_info.csv')
    # #%%
    # df = pd.read_csv('..//..//file_repo//info//dataset//dbc_shots_200pre//200pre_info.csv')
    # # new a column named test_auc
    # df['test_auc'] = 0
    # df.loc[0, 'test_auc'] = 0.5

