#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/27 0:19
# @Author  : zhongyu
# @Site    : 
# @File    : ip_concate_new_file.py

'''

from jddb.file_repo import FileRepo
import numpy as np
import os
from jddb.processor import ShotSet

if __name__ == '__main__':
    # %%
    # load file_repo
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'

    val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//'))
    train_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train//$shot_2$00//'))
    test_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))
    all_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_all//$shot_2$00//'))

    # %%
    # concate all shots as one file_repo
    test_shot_list = test_file_repo.get_all_shots()
    train_shot_list = train_file_repo.get_all_shots()
    val_shot_list = val_file_repo.get_all_shots()
    test_shotset = ShotSet(test_file_repo, test_shot_list)
    train_shotset = ShotSet(train_file_repo, train_shot_list)
    val_shotset = ShotSet(val_file_repo, val_shot_list)

    # %%
    # remove signal from all shots
    test__new_shotset = test_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                       shot_filter=test_shot_list,
                                                       save_repo=all_file_repo)
    train__new_shotset = train_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                         shot_filter=train_shot_list,
                                                         save_repo=all_file_repo)
    val__new_shotset = val_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                        shot_filter=val_shot_list,
                                                        save_repo=all_file_repo)
    #%%
    # spare dataset into train and test
    # shots list load
    shots_1 = np.load(os.path.join(dbc_data_dir, 'info//ip_info//ip_1.npy'))
    shots_2 = np.load(os.path.join(dbc_data_dir, 'info//ip_info//ip_2.npy'))
    shots_3 = np.load(os.path.join(dbc_data_dir, 'info//ip_info//ip_3.npy'))
    shots_4 = np.load(os.path.join(dbc_data_dir, 'info//ip_info//ip_4.npy'))
    shots_5 = np.load(os.path.join(dbc_data_dir, 'info//ip_info//ip_5.npy'))

    # concat shots 1-4 as training shots, shot 5 as test shots
    training_shots = np.concatenate((shots_1, shots_2, shots_3, shots_4))
    test_shots = shots_5

    all_shot_list = all_file_repo.get_all_shots()
    all_shotset = ShotSet(all_file_repo, all_shot_list)
    train_shotset = all_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                              shot_filter=training_shots,
                                              save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_ip_train//$shot_2$00//')))
    test_shotset = all_shotset.remove_signal(tags=['stacked_data', 'label'], keep=True,
                                                shot_filter=test_shots,
                                                save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_ip_test//$shot_2$00//')))
