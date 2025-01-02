#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/13 19:24
# @Author  : zhongyu
# @Site    : 
# @File    : concate_file_repo.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import os
from jddb.processor import ShotSet

if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    dbc_shots_info = pd.read_csv(os.path.join(dbc_data_dir, 'info//dbc_shots_info.csv'))
    val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//'))
    train_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train//$shot_2$00//'))
    test_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))

    # # %%
    # val_shot_list = val_file_repo.get_all_shots()
    # train_shot_list = train_file_repo.get_all_shots()
    # val_shotset = ShotSet(val_file_repo, val_shot_list)
    # train_shotset = ShotSet(train_file_repo, train_shot_list)
    # train_shotset = train_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True, shot_filter=train_shot_list,
    #                                             save_repo=FileRepo(
    #                                                 os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//')))
    # val_shotset = val_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True, shot_filter=val_shot_list,
    #                                         save_repo=FileRepo(
    #                                             os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//')))

    # %%
    # keep the shots ratio of test set is 0.21, take 100 shots from test set
    # the rest of test set as new test set
    test_shot_list = test_file_repo.get_all_shots()
    test_info = dbc_shots_info[dbc_shots_info['shot'].isin(test_shot_list)]
    test_info = test_info.sort_values(by='dbc')
    test_back_info = test_info.iloc[:100]
    test_back_shots = test_back_info['shot']
    test_shot_new_list = np.setdiff1d(test_shot_list, test_back_shots)
    # save test_back_shots and test_shot_new_list as npy
    np.save(os.path.join(dbc_data_dir, 'info//test_new_back//test_back_shots.npy'), test_back_shots)
    np.save(os.path.join(dbc_data_dir, 'info//test_new_back//test_new_shots.npy'), test_shot_new_list)

    test_new_shotset = ShotSet(test_file_repo, test_shot_new_list)
    test_back_shotset = ShotSet(test_file_repo, test_back_shots)
    test__new_shotset = test_new_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                       shot_filter=test_shot_new_list,
                                                       save_repo=FileRepo(
                                                           os.path.join(dbc_data_dir, 'label_test_new//$shot_2$00//')))
    test_back_shotset = test_back_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                        shot_filter=test_back_shots,
                                                        save_repo=FileRepo(
                                                            os.path.join(dbc_data_dir, 'label_train_val_no_sub')))
