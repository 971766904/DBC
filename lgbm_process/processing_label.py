#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/26 21:40
# @Author  : zhongyu
# @Site    : 
# @File    : processing_label.py

'''
from jddb.file_repo import FileRepo
import pandas as pd
import numpy as np
from jddb.processor import ShotSet
from jddb.processor.basic_processors import NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor
import json
import os

if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_lgbm'
    source_file_repo = FileRepo('..//..//file_repo//data_file//stack//$shot_2$00//')
    # shots info
    # %%
    # shots list load
    shots_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    shots_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    shots_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    shots_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')

    # concat shots 1-4 as training shots, shot 5 as test shots
    training_shots = np.concatenate((shots_1, shots_2, shots_3, shots_4, shots_5))
    test_shots = training_shots
    # %%
    #processing prepare
    shots_list = source_file_repo.get_all_shots()
    processed_shotset = ShotSet(source_file_repo, shots_list)

    # %%
    # spare dataset and record dbc

    # %%
    #
    # train_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=training_shots,
    #                                                 save_repo=FileRepo(
    #                                                     os.path.join(dbc_data_dir, 'remove_train//$shot_2$00//')))
    test_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=test_shots,
                                                   save_repo=FileRepo(
                                                       os.path.join(dbc_data_dir, 'remove_test//$shot_2$00//')))

    # %%
    # cut dataset
    # train_shotset = train_shotset.process(processor=CutProcessor(pre_time=20, is_test=False),
    #                                       input_tags=['stacked_data'],
    #                                       output_tags=['stacked_data'],
    #                                       shot_filter=training_shots,
    #                                       save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_train//$shot_2$00//')),
    #                                       processes=10)

    test_shotset = test_shotset.process(processor=CutProcessor(pre_time=20, is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['stacked_data'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_test//$shot_2$00//')),
                                        processes=10)

    # %%
    # label
    # train_shotset = train_shotset.process(processor=BinaryLabelProcessor(is_test=False),
    #                                       input_tags=['stacked_data'],
    #                                       output_tags=['label'],
    #                                       shot_filter=training_shots,
    #                                       save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_train_all//$shot_2$00//')),
    #                                       processes=10)

    test_shotset = test_shotset.process(processor=BinaryLabelProcessor(is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['label'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_test_all//$shot_2$00//')),
                                        processes=10)
