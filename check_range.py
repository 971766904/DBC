#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/12/26 21:25
# @Author  : zhongyu
# @Site    : 
# @File    : check_range.py

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor import Shot
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from sklearn.model_selection import train_test_split
import json

if __name__ == '__main__':
    # load file
    source_file_repo = FileRepo('..//file_repo//$shot_2$00//')
    shot_list = source_file_repo.get_all_shots()

    # read data & get info as numpy or dataframe
    # is disruption, ip, bt, Pin, Prad
    shots_info = pd.DataFrame(
        columns=['shot', 'IsDisrupt', 'Intentional_Disrupt','ip','bt','p'])
    for shot in shot_list:
        shot_info = {}
        shot_info['shot'] = shot
        lables = source_file_repo.read_labels(
            shot, ['IsDisrupt', 'DownTime', 'StartTime', 'Intentional_Disrupt'])
        shot_info['IsDisrupt'] = lables['IsDisrupt']
        shot_info['Intentional_Disrupt'] = lables['Intentional_Disrupt']
        data_raw = source_file_repo.read_data(shot, ['ip', 'P_in', 'P_rad', 'bt'])
        shot_info['ip'] = np.mean(data_raw['ip'])
        shot_info['p'] = np.mean(data_raw['P_in'] - data_raw['P_rad'])
        shot_info['bt'] = np.mean(data_raw['bt'])
        shots_info = shots_info.append(shot_info,ignore_index=True)
    dis_shots = shots_info.loc[
        (shots_info['IsDisrupt'] == True) & (shots_info['Intentional_Disrupt'] == False)]
    print(len(dis_shots))
    dis_shots.to_csv('..//file_repo//info//info.csv')

    # plot it
    plt.figure()
    dis_shots['ip'].plot.hist()
    plt.xlabel('ip/kA')
    plt.tight_layout()
    plt.savefig('..//file_repo//info//ip.png')

    plt.figure()
    dis_shots['bt'].plot.hist()
    plt.xlabel('bt/T')
    plt.tight_layout()
    plt.savefig('..//file_repo//info//bt.png')

    plt.figure()
    dis_shots['p'].plot.hist()
    plt.xlabel('p/kW')
    plt.tight_layout()
    plt.savefig('..//file_repo//info//p.png')

