#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2025/1/18 14:07
# @Author  : zhongyu
# @Site    :
# @File    : data_check.py

'''
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd

from jddb.performance import Result
from jddb.performance import Report

from jddb.file_repo import FileRepo

if __name__ == '__main__':
    #%%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file'
    dbc_info_dir = '..//..//file_repo//info'
    normalize_file_repo = FileRepo(os.path.join(dbc_data_dir, 'normalization//paper//$shot_2$00//'))
    feature_file_repo = FileRepo(os.path.join(dbc_data_dir, 'points_hdf5//feature//$shot_2$00//'))
    file_repo = normalize_file_repo
    # dbc_info = pd.read_csv(os.path.join(dbc_data_dir, 'info//std_info//shots_info.csv'))

    # chech the shots list and tags list of normalization file repo and feature file repo
    shots_list = file_repo.get_all_shots()
    tags = file_repo.get_tag_list(shots_list[0])
    shots_list_feature = feature_file_repo.get_all_shots()
    tags_feature = feature_file_repo.get_tag_list(shots_list_feature[0])
    # check if all the shots in feature file repo are in normalization file repo
    shots_diff = list(set(shots_list_feature) - set(shots_list))
    # check if all the tags in feature file repo are in normalization file repo
    tags_diff = list(set(tags_feature) - set(tags))

    #%%
    # load the shots info
    shots_list = file_repo.get_all_shots()
    # get tags
    tags = file_repo.get_tag_list(shots_list[0])
    #%%
    # create a dataframe to store the shot number, disruption, ipflattop and downtime
    shots_info = pd.DataFrame(columns=['shot', 'disruption', 'ipflat', 'downtime'])
    for shot in shots_list:
        shot_info = {}
        shot_info['shot'] = shot
        labels = file_repo.read_labels(shot, ['IsDisrupt', 'IpFlat', 'DownTime'])
        shot_info['disruption'] = labels['IsDisrupt']
        shot_info['IpFlat'] = labels['IpFlat']
        shot_info['downtime'] = labels['DownTime']
        shots_info = shots_info.append(shot_info, ignore_index=True)
    # #%%
    # # save the shots info
    # shots_info.to_csv(os.path.join(dbc_info_dir, 'shots_info.csv'), index=False)


