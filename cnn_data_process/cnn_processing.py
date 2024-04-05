#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/2 11:05
# @Author  : zhongyu
# @Site    : 
# @File    : cnn_processing.py

'''
from jddb.file_repo import FileRepo
import pandas as pd
from jddb.processor import ShotSet
from jddb.processor.basic_processors import NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, SliceProcessor, StackProcessor,OldSliceProcessor
import json
import os

if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file'
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    shots_info = pd.read_csv('..//..//file_repo//info//std_info//shots_info.csv')
    train_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_cnn//$shot_2$00//')
    shots_list = shots_info['shot']
    source_shotset = ShotSet(source_file_repo, shots_list)
    tag_list = source_file_repo.get_tag_list(shots_list[0])
    tags_sxr = find_tags('sxr', tag_list)
    tags_axuv = find_tags('AXUV', tag_list)
    basic_tags = ["ip", "bt", "vl", "dx", "dy",
                  "polaris_den_v01", "polaris_den_v09", "polaris_den_v17",
                  'P_in', 'P_rad', 'ip_error', 'n=1 amplitude', 'ne0', 'ne_nG',
                  'qa_proxy', 'radiation_proxy', 'rotating_mode_proxy']
    target_tags = basic_tags + tags_sxr + tags_axuv

    # # %%
    # processed_shotset = source_shotset.remove_signal(tags=target_tags, keep=True,
    #                                                  save_repo=train_file_repo)
    #
    # # %%
    # # 1. normalize data
    # # basic signal
    # normalization_param = read_config('config/normalization_params.json')
    # for tag in basic_tags:
    #     mean = normalization_param[tag][0]
    #     std = normalization_param[tag][1]
    #     processed_shotset = processed_shotset.process(
    #         processor=NormalizationProcessor(mean=mean, std=std),
    #         input_tags=[tag],
    #         output_tags=[tag],
    #         save_repo=train_file_repo,
    #         processes=10)
    # # sxr array
    # mean = normalization_param['sxr'][0]
    # std = normalization_param['sxr'][1]
    # for tag in tags_sxr:
    #     processed_shotset = processed_shotset.process(
    #         processor=NormalizationProcessor(mean=mean, std=std),
    #         input_tags=[tag],
    #         output_tags=[tag],
    #         save_repo=train_file_repo,
    #         processes=10)
    # # AXUV array
    # mean = normalization_param['AXUV'][0]
    # std = normalization_param['AXUV'][1]
    # for tag in tags_axuv:
    #     processed_shotset = processed_shotset.process(
    #         processor=NormalizationProcessor(mean=mean, std=std),
    #         input_tags=[tag],
    #         output_tags=[tag],
    #         save_repo=train_file_repo,
    #         processes=10)
    #
    # # %%
    # # 2. slice data
    # print('slice...')
    # processed_shotset = ShotSet(train_file_repo, shots_list)
    # processed_shotset = processed_shotset.process(
    #     TrimProcessor(),
    #     input_tags=[target_tags],
    #     output_tags=[target_tags],
    #     save_repo=FileRepo(os.path.join(dbc_data_dir, 'trim//$shot_2$00//')),
    #     processes=10)
    # # os.makedirs(os.path.join(dbc_data_dir, 'slice'), exist_ok=True)
    #
    #
    # # %%
    # # 3. trim data
    # processed_shotset = processed_shotset.process(
    #     processor=StackProcessor(),
    #     input_tags=[target_tags],
    #     output_tags=['stacked_data'],
    #     save_repo=FileRepo(os.path.join(dbc_data_dir, 'stack//$shot_2$00//')),
    #     processes=10)

    # %%
    # 4. stack features to matrix
    processed_shotset = ShotSet(FileRepo(os.path.join(dbc_data_dir, 'stack//$shot_2$00//')), shots_list)
    processed_shotset = processed_shotset.process(
        processor=SliceProcessor(32, 31 / 32),
        input_tags=['stacked_data'],
        output_tags=['stacked_data'],
        save_repo=FileRepo(os.path.join(dbc_data_dir, 'slice//$shot_2$00//')),
        processes=10)
