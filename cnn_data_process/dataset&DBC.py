#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/12 16:41
# @Author  : zhongyu
# @Site    : 
# @File    : dataset&DBC.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import os

if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix'
    source_file_repo = FileRepo('..//..//file_repo//data_file//slice//$shot_2$00//')
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//training_shots.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//valid_shots.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')

    # %%
    # calculate total dbc of training shots, valid shots and test shots
    # according to the dbc value of each shot in dbc_shots_info which is the column 'dbc',
    # and the shots in training_shots, valid_shots and test_shots
    training_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(training_shots)]['dbc'].sum()
    valid_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(valid_shots)]['dbc'].sum()
    test_dbc = dbc_shots_info[dbc_shots_info['shot'].isin(test_shots)]['dbc'].sum()

    # %%
    # create new training shots, valid shots from the original training shots, valid shots
    # the new training shots should have 7000 total dbc
    # the new valid shots should have 2000 total dbc
    # Combine original training, validation, and test shots
    combined_shots = np.concatenate((training_shots, valid_shots))

    # Filter dbc_shots_info to only include shots in combined_shots
    dbc_shots_info_combined = dbc_shots_info[dbc_shots_info['shot'].isin(combined_shots)]
    dis_shots_info = dbc_shots_info_combined[dbc_shots_info_combined['IsDisrupt'] == 1]
    undis_shots_info = dbc_shots_info_combined[dbc_shots_info_combined['IsDisrupt'] == 0]

    # #%%
    # # plan A: in the order of shots in dis_shots_info and undis_shots_info , add shots to training_shots until
    # # the total dbc of training_shots is 7000
    # # the ratio for dis shots should be considered
    # training_dbc = 0
    # training_shots = []
    # for index, row in dis_shots_info.iterrows():
    #     if training_dbc + row['dbc'] <= 7000:
    #         training_dbc += row['dbc']
    #         training_shots.append(row['shot'])
    # for index, row in undis_shots_info.iterrows():
    #     if training_dbc + row['dbc'] <= 7000:
    #         training_dbc += row['dbc']
    #         training_shots.append(row['shot'])

    # %%
    # plan B:
    wrong_predict_shots_1 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots.npy')
    wrong_predict_shots_2 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots.npy')

    train_val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//'))
    # wrong_predict_shots_1 is subset of valid shots, now add it to training shots and eliminate it from valid shots
    training_shots = np.concatenate((training_shots, wrong_predict_shots_1))
    valid_shots = np.setdiff1d(valid_shots, wrong_predict_shots_1)
    # wrong_predict_shots_2 is subset of training shots, now eliminate it from training shots
    training_shots = np.setdiff1d(training_shots, wrong_predict_shots_2)
    # save the new training shots and valid shots
    np.save('..//..//file_repo//info//dataset//training_shots_1.npy', training_shots)
    np.save('..//..//file_repo//info//dataset//valid_shots_1.npy', valid_shots)
    train_shotset = train_val_file_repo.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                      shot_filter=training_shots,
                                                      save_repo=FileRepo(
                                                          os.path.join(dbc_data_dir, 'label_train//$shot_2$00//')))
    val_shotset = train_val_file_repo.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                    shot_filter=valid_shots,
                                                    save_repo=FileRepo(
                                                        os.path.join(dbc_data_dir, 'label_val//$shot_2$00//')))
