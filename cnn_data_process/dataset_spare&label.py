#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/2 16:33
# @Author  : zhongyu
# @Site    : 
# @File    : dataset_spare&label.py

'''
from jddb.file_repo import FileRepo
import pandas as pd
import numpy as np
from jddb.processor import ShotSet
import os
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor

if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn'
    source_file_repo = FileRepo('..//..//file_repo//data_file//slice//$shot_2$00//')
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')
    training_shots = np.load('..//..//file_repo//info//dataset//training_shots.npy')
    valid_shots = np.load('..//..//file_repo//info//dataset//valid_shots.npy')
    test_shots = np.load('..//..//file_repo//info//dataset//test_shots.npy')
    shots_list = source_file_repo.get_all_shots()
    processed_shotset = ShotSet(source_file_repo, shots_list)

    # %%
    # spare dataset and record dbc

    # %%
    #
    train_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=training_shots,
                                                    save_repo=FileRepo(
                                                        os.path.join(dbc_data_dir, 'remove_train//$shot_2$00//')))
    val_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=valid_shots,
                                                  save_repo=FileRepo(
                                                      os.path.join(dbc_data_dir, 'remove_val//$shot_2$00//')))
    test_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=test_shots,
                                                   save_repo=FileRepo(
                                                       os.path.join(dbc_data_dir, 'remove_test//$shot_2$00//')))

    # %%
    # cut dataset
    train_shotset = train_shotset.process(processor=CutProcessor(pre_time=20, is_test=False),
                                          input_tags=['stacked_data'],
                                          output_tags=['stacked_data'],
                                          shot_filter=training_shots,
                                          save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_train//$shot_2$00//')),
                                          processes=10)
    val_shotset = val_shotset.process(processor=CutProcessor(pre_time=20, is_test=False),
                                      input_tags=['stacked_data'],
                                      output_tags=['stacked_data'],
                                      shot_filter=valid_shots,
                                      save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_val//$shot_2$00//')),
                                      processes=10)
    test_shotset = test_shotset.process(processor=CutProcessor(pre_time=20, is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['stacked_data'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_test//$shot_2$00//')),
                                        processes=10)

    # %%
    # label
    train_shotset = train_shotset.process(processor=BinaryLabelProcessor(is_test=False),
                                          input_tags=['stacked_data'],
                                          output_tags=['label'],
                                          shot_filter=training_shots,
                                          save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_train//$shot_2$00//')),
                                          processes=10)
    val_shotset = val_shotset.process(processor=BinaryLabelProcessor(is_test=False),
                                      input_tags=['stacked_data'],
                                      output_tags=['label'],
                                      shot_filter=valid_shots,
                                      save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//')),
                                      processes=10)
    test_shotset = test_shotset.process(processor=BinaryLabelProcessor(is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['label'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_test//$shot_2$00//')),
                                        processes=10)
