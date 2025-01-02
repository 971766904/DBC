#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/13 21:30
# @Author  : zhongyu
# @Site    : 
# @File    : plan_b_dataset_construct.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import os
from jddb.processor import ShotSet, Shot
from util.basic_processor import CutProcessor

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
    wrong_predict_shots_1 = np.load('..//..//file_repo//info//dataset//wrong_predict_shots.npy')

    train_val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//'))
    train_val_shot_list = train_val_file_repo.get_all_shots()
    train_val_shotset = ShotSet(train_val_file_repo, train_val_shot_list)
    # wrong_predict_shots_1 is subset of valid shots, now add it to training shots and eliminate it from valid shots
    training_shots = np.concatenate((training_shots, wrong_predict_shots_1))
    valid_shots = np.setdiff1d(valid_shots, wrong_predict_shots_1)
    # save the new training shots and valid shots
    np.save('..//..//file_repo//info//dataset//training_shots_1.npy', training_shots)
    np.save('..//..//file_repo//info//dataset//valid_shots_1.npy', valid_shots)

    train_shotset = train_val_shotset.process(processor=CutProcessor(pre_time=5, is_test=False),
                                              input_tags=['label'],
                                              output_tags=['trans'],
                                              shot_filter=training_shots,
                                              save_repo=FileRepo(
                                                  os.path.join(dbc_data_dir, 'label_train//$shot_2$00//')),
                                              processes=10)
    val_shotset = train_val_shotset.process(processor=CutProcessor(pre_time=5, is_test=False),
                                            input_tags=['label'],
                                            output_tags=['trans'],
                                            shot_filter=valid_shots,
                                            save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//')),
                                            processes=10)
    # %%
    for shot in training_shots:
        shot_ob = Shot(shot, train_val_file_repo)
        shot_ob.save(FileRepo(os.path.join(dbc_data_dir, 'label_train//$shot_2$00//')), data_type=float)
    for shot in valid_shots:
        shot_ob = Shot(shot, train_val_file_repo)
        shot_ob.save(FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//')), data_type=float)
