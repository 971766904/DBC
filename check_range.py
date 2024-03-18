#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/12/26 21:25
# @Author  : zhongyu
# @Site    : 
# @File    : check_range.py

'''
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo
from util.read_data import read_data_from_tree
from jddb.processor import ShotSet
from jddb.processor import Shot
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from sklearn.model_selection import train_test_split
import json


def cq_time_cal(shot, downtime):
    ip_data, ip_time = read_data_from_tree(shot, r'\ip')

    ip_df = pd.DataFrame(columns=['ip', 'time'])
    ip_df['ip'] = ip_data
    ip_df['time'] = ip_time
    pre_dis = ip_df.loc[(downtime > ip_df['time']) & (ip_df['time'] > downtime - 0.03)]
    tq_data = ip_df.loc[(downtime + 0.02 > ip_df['time']) & (ip_df['time'] > downtime)]
    pre_dis_ip_mean = pre_dis['ip'].mean()
    tq_cal_range = tq_data.loc[(pre_dis_ip_mean * 0.8 > tq_data['ip']) & (tq_data['ip'] > pre_dis_ip_mean * 0.2)]
    tq_time = tq_cal_range.iloc[-1]['time'] - tq_cal_range.iloc[0]['time']
    cq_rate = pre_dis_ip_mean * 0.6 / tq_time
    return tq_time, cq_rate


if __name__ == '__main__':
    # %%
    # load file
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    shot_list = source_file_repo.get_all_shots()
    valid_shots = np.load('..//file_repo//info//std_info//used_shots.npy')

    # %%
    # read data & get info as numpy or dataframe
    # is disruption, ip, bt, Pin, Prad, duration, CQ rate(disruption only)
    shots_info = pd.DataFrame(
        columns=['shot', 'IsDisrupt', 'Intentional_Disrupt', 'duration',
                 'CQ_time', 'CQ_rate', 'ip', 'bt', 'p'])
    for shot in valid_shots:
        shot_info = {'shot': shot}
        lables = source_file_repo.read_labels(
            shot, ['IsDisrupt', 'DownTime', 'StartTime', 'Intentional_Disrupt'])
        shot_info['IsDisrupt'] = lables['IsDisrupt']
        shot_info['Intentional_Disrupt'] = lables['Intentional_Disrupt']
        data_raw = source_file_repo.read_data(shot, ['ip', 'P_in', 'P_rad', 'bt'])
        shot_info['ip'] = np.mean(data_raw['ip'])
        shot_info['p'] = np.mean(data_raw['P_in'] - data_raw['P_rad'])
        shot_info['bt'] = np.mean(data_raw['bt'])
        shot_info['duration'] = lables['DownTime'] - lables['StartTime']
        if lables['IsDisrupt']:
            shot_info['CQ_time'], shot_info['CQ_rate'] = cq_time_cal(shot, lables['DownTime'])
        else:
            shot_info['CQ_time'] = 0
            shot_info['CQ_rate'] = 0
        shots_info = shots_info.append(shot_info, ignore_index=True)
    sta_des = shots_info.describe()
    sta_des.to_csv('..//file_repo//info//std_info//des_statistics_tq.csv', index=True)
    dis_shot = shots_info.loc[(shots_info['IsDisrupt'] == True)]
    dis_shot['CQ_time'] = dis_shot['CQ_time'].astype(float)
    dis_des = dis_shot.describe()
    dis_des.to_csv('..//file_repo//info//std_info//disruption_statistics_tq.csv', index=True)
    shots_info.to_csv('..//file_repo//info//std_info//shots_info.csv', index=False)
    print(len(shots_info))
    # dis_shots.to_csv('..//file_repo//info//info.csv')

    # # %%
    # # plot it
    # plt.figure()
    # shots_info['ip'].plot.hist()
    # plt.xlabel('ip/kA')
    # plt.tight_layout()
    # # plt.savefig('..//file_repo//info//ip.png')
    #
    # plt.figure()
    # shots_info['bt'].plot.hist()
    # plt.xlabel('bt/T')
    # plt.tight_layout()
    # # plt.savefig('..//file_repo//info//bt.png')
    #
    # plt.figure()
    # shots_info['p'].plot.hist()
    # plt.xlabel('p/kW')
    # plt.tight_layout()
    # # plt.savefig('..//file_repo//info//p.png')
