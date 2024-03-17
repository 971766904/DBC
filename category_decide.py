#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/17 17:20
# @Author  : zhongyu
# @Site    : 
# @File    : category_decide.py

'''
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo
from util.read_data import read_data_from_tree

if __name__ == '__main__':
    # %%
    # load file
    valid_shots = np.load('..//file_repo//info//std_info//used_shots.npy')
    shots_info = pd.read_csv('..//file_repo//info//std_info//shots_info.csv')

    # %%
    # classifying according to both disruption and undis shots
    cate_1 = shots_info.loc[(shots_info['ip'] >= 181.5168)
                            & (shots_info['bt'] >= 1.99084)
                            & (shots_info['p'] >= 320.755)]
    cate_1_left = shots_info.loc[~((shots_info['ip'] >= 181.5168)
                                   & (shots_info['bt'] >= 1.99084)
                                   & (shots_info['p'] >= 320.755))]
    cate_2 = cate_1_left.loc[(cate_1_left['ip'] >= 162.8567)
                             & (cate_1_left['bt'] >= 1.791811)
                             & (cate_1_left['p'] >= 278.0338)]
    cate_2_left = cate_1_left.loc[~((cate_1_left['ip'] >= 162.8567)
                                    & (cate_1_left['bt'] >= 1.791811)
                                    & (cate_1_left['p'] >= 278.0338))]
    cate_3 = cate_2_left.loc[(cate_2_left['ip'] >= 149.935)
                             & (cate_2_left['bt'] >= 1.691232)
                             & (cate_2_left['p'] >= 238.4405)]
    cate_4 = cate_2_left.loc[~((cate_2_left['ip'] >= 149.935)
                               & (cate_2_left['bt'] >= 1.691232)
                               & (cate_2_left['p'] >= 238.4405))]
