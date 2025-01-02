#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/28 15:00
# @Author  : zhongyu
# @Site    : 
# @File    : get_shots_for_balance.py

'''
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo
from scipy.signal import find_peaks, savgol_filter, stft

if __name__ == '__main__':
    # %%
    # load csv file
    shots_info = pd.read_csv('..//file_repo//info//std_info//shots_info.csv')
    stat_info = pd.read_csv('..//file_repo//info//std_info//des_statistics_tq.csv')
    dis_stat_info = pd.read_csv('..//file_repo//info//std_info//disruption_statistics_tq.csv')
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')

    # %%
    #
    shots_efit = pd.DataFrame(
        columns=['shot', 'sawtooth', 'ip', 'bt', 'efit time'])
    shot = 1051984
    # %%
    # for shot in shots_info['shot']:
    #     data_sxr = source_file_repo.read_data(shot, ["sxr_cb_032"])
    #     t_start = source_file_repo.read_labels(shot, ['StartTime'])
    #     t = t_start['StartTime'] + np.arange(data_sxr["sxr_cb_032"].shape[0]) * 0.001
    #     plt.figure()
    #     ax1 = plt.subplot(111)
    #     ax1.plot(t, data_sxr["sxr_cb_032"], 'r')
    #     plt.title('disruption or not:{}'.format(shots_info[shots_info['shot'] == shot]['IsDisrupt'].values))
    #     plt.tight_layout()
    #     # plt.savefig('./_temp_fig/{}.png'.format(shot))
    #     plt.close()

    # %%
    # calculate dbc, which is the sum of three standardization. If the shot is disruption, add CQ rate to the sum
    # and multiply 3 as the weight of CQ rate
    ip_min = stat_info['ip'][3]
    bt_min = stat_info['bt'][3]
    p_min = stat_info['p'][3]
    ip_std = stat_info['ip'][2]
    bt_std = stat_info['bt'][2]
    p_std = stat_info['p'][2]
    cq_rate_min = dis_stat_info['CQ_rate'][3]
    cq_rate_std = dis_stat_info['CQ_rate'][2]
    shots_info['dbc'] = shots_info.apply(
        lambda row: (((row['ip'] - ip_min) / ip_std) ** 2 + ((row['bt'] - bt_min) / bt_std) ** 2
                     + ((row['p'] - p_min) / p_std) ** 2 + ((row['CQ_rate'] - cq_rate_min) / cq_rate_std) ** 6)
        if row['IsDisrupt'] else (((row['ip'] - ip_min) / ip_std) ** 2 + ((row['bt'] - bt_min) / bt_std) ** 2
                                  + ((row['p'] - p_min) / p_std) ** 2) * 0.21, axis=1)

    # shots_info.to_csv('..//file_repo//info//std_info//dbc_shots_info.csv', index=False)

    # %%
    #
    shots_info['IsDisrupt'] = shots_info['IsDisrupt'].map({'True': True, '1': True, 'False': False})
    shots_info.to_csv('..//file_repo//info//std_info//power_dbc_shots_info.csv', index=False)
    for index, row in shots_info.iterrows():
        shot = row['shot']
        sawtooth = 0
        ip = row['ip']
        bt = row['bt']
        efit_time = source_file_repo.read_labels(shot, ['DownTime'])['DownTime'] - 0.04
        shots_efit = shots_efit.append({'shot': shot, 'sawtooth': sawtooth, 'ip': ip, 'bt': bt,
                                        'efit time': efit_time}, ignore_index=True)
    # shots_efit.to_csv('..//file_repo//info//std_info//efit_shots_info.csv', index=False)

    # %%
    # process shots_info
    # it return keyerror:'ip'
    # show me why

    shots_info['n_ip'] = (shots_info['ip'] - ip_min) / ip_std
    shots_info['n_bt'] = (shots_info['bt'] - bt_min) / bt_std
    shots_info['n_p'] = (shots_info['p'] - p_min) / p_std
    # describe the shots_info
    #%%
    shots_info_des=shots_info.describe()
    #%%
    # plot the distribution of shots_info
    plt.figure()
    plt.hist(shots_info['n_ip'], bins=100, alpha=0.5, label='ip')
    plt.hist(shots_info['n_bt'], bins=100, alpha=0.5, label='bt')
    plt.hist(shots_info['n_p'], bins=100, alpha=0.5, label='p')
    plt.legend()
    plt.show()

