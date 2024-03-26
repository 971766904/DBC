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
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    valid_shots = np.load('..//file_repo//info//std_info//used_shots.npy')
    shots_info = pd.read_csv('..//file_repo//info//std_info//shots_info.csv')
    # the column 'IsDisrupt' is object, which should be mapped to bool
    shots_info['IsDisrupt'] = shots_info['IsDisrupt'].map({'True': True, '1': True, 'False': False})

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
    cate_1.to_csv('..//file_repo//info//std_info//category//cate_1.csv', index=False)
    cate_2.to_csv('..//file_repo//info//std_info//category//cate_2.csv', index=False)
    cate_3.to_csv('..//file_repo//info//std_info//category//cate_3.csv', index=False)
    cate_4.to_csv('..//file_repo//info//std_info//category//cate_4.csv', index=False)

    # %%
    # add category to hdf5 file by adding a metadata
    # cate 1 contains 39 shots with 10 disruption
    shots_1 = cate_1['shot']
    for shot in shots_1:
        source_file_repo.write_label(shot, {'category': 1})
    # cate 2 contains 312 shots with 40 disruption
    shots_2 = cate_2['shot']
    for shot in shots_2:
        source_file_repo.write_label(shot, {'category': 2})
    # cate 1 contains 513 shots with 103 disruption
    shots_3 = cate_3['shot']
    for shot in shots_3:
        source_file_repo.write_label(shot, {'category': 3})
    # cate 4 contains 827 shots with 219 disruption
    shots_4 = cate_4['shot']
    for shot in shots_4:
        source_file_repo.write_label(shot, {'category': 4})
