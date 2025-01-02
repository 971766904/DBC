#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/26 23:10
# @Author  : zhongyu
# @Site    : 
# @File    : _temp_check_shots.py

'''
from jddb.file_repo import FileRepo
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # load file repo
    source_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_cnn//all_mix//label_train+val//$shot_2$00//')

    # load shots info
    shots_1 = np.load('..//..//file_repo//info//ip_info//ip_1.npy')
    shots_2 = np.load('..//..//file_repo//info//ip_info//ip_2.npy')
    shots_3 = np.load('..//..//file_repo//info//ip_info//ip_3.npy')
    shots_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    shots_info = pd.read_csv('..//..//file_repo//info//std_info//shots_info.csv')
    # concat shots 1-4 as training shots, shot 5 as test shots
    training_shots = np.concatenate((shots_1, shots_2, shots_3, shots_4))

    # # check the shots  info is in the file_repo
    # for shot in source_file_repo.get_all_shots():
    #     if shot not in training_shots:
    #         print(f'{shot} not in the file_repo')

