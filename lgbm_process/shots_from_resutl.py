#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/2 22:55
# @Author  : zhongyu
# @Site    : 
# @File    : shots_from_resutl.py

'''
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # read the prediction result
    shot_result = pd.read_csv('..//..//file_repo//ip_eval//case1_11_27/test_result.csv')
    # get the shots which have false positive or false negative equal to 1
    shot_result = shot_result[(shot_result['false_positive'] == 1) | (shot_result['false_negative'] == 1)]
    # get the shots_no
    shots_no = shot_result['shot_no'].values
    # save the shots_no
    np.save('..//..//file_repo//info//ip_info//test_shots.npy', shots_no)
