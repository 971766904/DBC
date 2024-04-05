#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/2 15:15
# @Author  : zhongyu
# @Site    : 
# @File    : mean_std_calculate.py

'''
from jddb.file_repo import FileRepo
import pandas as pd
import numpy as np
from util.basic_processor import find_tags
from sklearn.preprocessing import StandardScaler
import json

if __name__ == '__main__':
    # %%
    # load file repo#
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    shots_info = pd.read_csv('..//..//file_repo//info//std_info//shots_info.csv')
    shots_list = shots_info['shot']
    tag_list = source_file_repo.get_tag_list(shots_list[0])
    tags_sxr = find_tags('sxr', tag_list)
    tags_axuv = find_tags('AXUV', tag_list)
    basic_tags = ["ip", "bt", "vl", "dx", "dy",
                  "polaris_den_v01", "polaris_den_v09", "polaris_den_v17",
                  'P_in', 'P_rad', 'ip_error', 'n=1 amplitude', 'ne0', 'ne_nG',
                  'qa_proxy', 'radiation_proxy', 'rotating_mode_proxy']

    # %%
    # initialization
    normalization_dic = dict()
    data_cal = dict()
    data_cal['sxr'] = np.empty(0)
    data_cal['AXUV'] = np.empty(0)
    for tag in basic_tags:
        data_cal[tag] = np.empty(0)

    #%%
    # calculate by shot
    for shot in shots_list:
        # sxr array calculate
        sxr_array_data = source_file_repo.read_data(shot, tags_sxr)
        for tag in tags_sxr:
            data_cal['sxr'] = np.concatenate((data_cal['sxr'], sxr_array_data[tag]))
        # AXUV array calculate
        AXUV_array_data = source_file_repo.read_data(shot, tags_axuv)
        for tag in tags_axuv:
            data_cal['AXUV'] = np.concatenate((data_cal['AXUV'], AXUV_array_data[tag]))
        # basic calculate
        basic_data = source_file_repo.read_data(shot, basic_tags)
        for tag in basic_tags:
            data_cal[tag] = np.concatenate((data_cal[tag], basic_data[tag]))
    sxr_scaler = StandardScaler()
    sxr_scaler.fit(data_cal['sxr'].reshape(-1, 1))
    normalization_dic['sxr'] = [sxr_scaler.mean_.tolist(), np.sqrt(sxr_scaler.var_).tolist()]
    AXUV_scaler = StandardScaler()
    AXUV_scaler.fit(data_cal['AXUV'].reshape(-1, 1))
    normalization_dic['AXUV'] = [AXUV_scaler.mean_.tolist(), np.sqrt(AXUV_scaler.var_).tolist()]
    for tag in basic_tags:
        scaler = StandardScaler()
        scaler.fit(data_cal[tag].reshape(-1, 1))
        normalization_dic[tag] = [scaler.mean_.tolist(), np.sqrt(scaler.var_).tolist()]

    # save normalization data
    json_file_path = 'config/normalization_params.json'
    json_data = json.dumps(normalization_dic, indent=4)  # indent参数用于指定缩进空格数，使JSON文件更易读
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
    print(f'Data has been saved to {json_file_path}')
