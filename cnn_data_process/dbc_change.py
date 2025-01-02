#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/14 19:45
# @Author  : zhongyu
# @Site    : 
# @File    : dbc_change.py

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
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//training_shots.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//valid_shots.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')
    original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    print('training_shots:', len(training_shots))
    print('valid_shots:', len(valid_shots))

    # # %%
    # # plan B:
    # wrong_predict_shots_2 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots_train_2.npy')
    # wrong_predict_shots_1 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots_val_2.npy')
    #
    # #%%
    # # exchange the shots between
    # # training shots and valid shots, change 10 disruption shots and 10 undistruption shots based on
    # # dbc of each shot
    # training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    # valid_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]
    # # test_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(test_shots)]
    # training_disruption_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 1].sort_values(by='dbc',ascending=False)
    # training_undisruption_shots = training_shots_dbc[training_shots_dbc['IsDisrupt'] == 0].sort_values(by='dbc',ascending=False)
    # valid_disruption_shots = valid_shots_dbc[valid_shots_dbc['IsDisrupt'] == 1].sort_values(by='dbc')
    # valid_undisruption_shots = valid_shots_dbc[valid_shots_dbc['IsDisrupt'] == 0].sort_values(by='dbc')
    # # choose 10 subset of disruption shots from training disruption shots, each subset has 10 shots, and
    # # they have different total dbc
    # training_disruption_shots_10 = []
    # for i in range(10):
    #     training_disruption_shots_10.append(training_disruption_shots.iloc[i*10:(i+1)*10])
    # # choose 10 subset of undisruption shots from training undisruption shots, each subset has 10 shots, and
    # # they have different total dbc
    # training_undisruption_shots_10 = []
    # for i in range(10):
    #     training_undisruption_shots_10.append(training_undisruption_shots.iloc[i*10:(i+1)*10])
    # # choose 10 subset of disruption shots from valid disruption shots, each subset has 10 shots, and
    # # they have different total dbc
    # valid_disruption_shots_10 = []
    # for i in range(10):
    #     valid_disruption_shots_10.append(valid_disruption_shots.iloc[i*10:(i+1)*10])
    # # choose 10 subset of undisruption shots from valid undisruption shots, each subset has 10 shots, and
    # # they have different total dbc
    # valid_undisruption_shots_10 = []
    # for i in range(10):
    #     valid_undisruption_shots_10.append(valid_undisruption_shots.iloc[i*10:(i+1)*10])
    # # exchange the shots between training shots and valid shots to get 10 different new training shots and valid shots,
    # # then eliminate these 10 shots from original training shots and valid shots
    # for i in range(10):
    #     print('====================')
    #     training_shots = np.setdiff1d(training_shots, training_disruption_shots_10[i]['shot'])
    #     training_shots = np.setdiff1d(training_shots, training_undisruption_shots_10[i]['shot'])
    #     valid_shots = np.setdiff1d(valid_shots, valid_disruption_shots_10[i]['shot'])
    #     valid_shots = np.setdiff1d(valid_shots, valid_undisruption_shots_10[i]['shot'])
    #     training_shots = np.concatenate((training_shots, valid_disruption_shots_10[i]['shot']))
    #     training_shots = np.concatenate((training_shots, valid_undisruption_shots_10[i]['shot']))
    #     valid_shots = np.concatenate((valid_shots, training_disruption_shots_10[i]['shot']))
    #     valid_shots = np.concatenate((valid_shots, training_undisruption_shots_10[i]['shot']))
    #     print('training_shots:', len(training_shots))
    #     print('valid_shots:', len(valid_shots))
    #     # calculate the total dbc of new training shots and valid shots
    #     training_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]
    #     valid_shots_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]
    #     training_total_dbc = training_shots_dbc['dbc'].sum()
    #     valid_total_dbc = valid_shots_dbc['dbc'].sum()
    #     print('training_total_dbc:', training_total_dbc)
    #     print('valid_total_dbc:', valid_total_dbc)
    #     print('----train shots:')
    #     original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
    #     # check if training_shots have the repeated shots
    #     if len(training_shots) != len(set(training_shots)):
    #         print('-----repeated shots in training_shots')
    #     # check if valid_shots have the repeated shots
    #     if len(valid_shots) != len(set(valid_shots)):
    #         print('-----repeated shots in valid_shots')
    #     # # save the new training shots and valid shots with different file name
    #     # np.save('..//..//file_repo//info//dataset//dbc_down//training_shots_{}.npy'.format(i+1), training_shots)
    #     # np.save('..//..//file_repo//info//dataset//dbc_down//valid_shots_{}.npy'.format(i+1), valid_shots)

    #%%
    # display the information of each new training shots and valid shots
    df = pd.DataFrame(columns=['original_dbc', 'dis_ratio', 'num_train'])
    for i in range(5):
        print('====================')
        print('Iteration:', i + 1)
        training_shots = np.load('..//..//file_repo//info//dataset//random_827//training_shots_{}.npy'.format(i+1))
        valid_shots = np.load('..//..//file_repo//info//dataset//random_827//valid_shots_{}.npy'.format(i+1))
        print('train shots:')
        original_dbc, training_dis_ratio, num_train = get_dataset_info(dbc_shots_info, training_shots)
        df.loc[i] = [original_dbc, training_dis_ratio, num_train]
        print('valid shots:')
        original_dbc, valid_dis_ratio, num_valid = get_dataset_info(dbc_shots_info, valid_shots)
        # check if training_shots have the repeated shots
        if len(training_shots) != len(set(training_shots)):
            print('-----repeated shots in training_shots')
        # check if valid_shots have the repeated shots
        if len(valid_shots) != len(set(valid_shots)):
            print('-----repeated shots in valid_shots')
    print(df)
    df.to_csv('..//..//file_repo//info//dataset//random_827//power_dbc_info.csv')
