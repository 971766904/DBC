#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/19 10:02
# @Author  : zhongyu
# @Site    : 
# @File    : dataset_shots.py

'''
# %%
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo

if __name__ == '__main__':
    # %%
    # load csv file
    cate_1 = pd.read_csv('..//file_repo//info//std_info//category//cate_1.csv')
    cate_2 = pd.read_csv('..//file_repo//info//std_info//category//cate_2.csv')
    cate_3 = pd.read_csv('..//file_repo//info//std_info//category//cate_3.csv')
    cate_4 = pd.read_csv('..//file_repo//info//std_info//category//cate_4.csv')

    # %%
    # shots arrange for train, validation and test
    shots1 = cate_1['shot']
    shots2 = cate_2['shot']
    shots3 = cate_3['shot']
    shots4 = cate_4['shot']
    test_set = np.array(pd.concat([shots1, shots2]))
    valid_set = np.array(shots3)
    training_set = np.array(shots4)
    np.save('..//file_repo//info//dataset//training_shots.npy', training_set)
    np.save('..//file_repo//info//dataset//valid_shots.npy', valid_set)
    np.save('..//file_repo//info//dataset//test_shots.npy', test_set)
