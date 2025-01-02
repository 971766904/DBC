#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/4 15:07
# @Author  : zhongyu
# @Site    : 
# @File    : dataset_test_compare_shots.py

'''
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os


from jddb.performance import Result
from jddb.performance import Report

from jddb.file_repo import FileRepo

if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_lgbm'
    file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train_all//$shot_2$00//'))
    # load the shots info
    shots_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    shots_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    shots_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    shots_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    # new a train set and test set from shots_1-4
    all_shots = np.concatenate((shots_1, shots_2, shots_3, shots_4))
    is_disrupt = []
    for shot in all_shots:
        # if shot is not int, change it to int
        shot = int(shot)
        dis_label = file_repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])
    train_shots, test_shots, _, _ = \
        train_test_split(all_shots, all_shots, test_size=0.2,
                         random_state=1, shuffle=True, stratify=is_disrupt)
    # %%
    #save the train and test shots
    np.save('..//..//file_repo//info//ip_info//randm_train_shots.npy', train_shots)
    np.save('..//..//file_repo//info//ip_info//randm_test_shots.npy', test_shots)
