#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/12 10:18
# @Author  : zhongyu
# @Site    : 
# @File    : shot_select & signal_plot.py

'''
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet

if __name__ == '__main__':
    # choose shots depending on filter and plot histogram of each signal, get the std and mean.
    # %%
    # load data
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    source_data = ShotSet(source_file_repo)
    shot_list = source_data.shot_list

    # %%
    # duration time scan
    duration_info = pd.DataFrame(columns=['shot', 'duration'])
    valid_shots = []
    for shot in shot_list:
        shot_info = {}
        lables = source_file_repo.read_labels(
            shot, ['IsDisrupt', 'DownTime', 'StartTime', 'Intentional_Disrupt'])
        shot_info['shot'] = shot
        shot_info['duration'] = lables['DownTime'] - lables['StartTime']
        duration_info = duration_info.append(shot_info, ignore_index=True)
        if shot_info['duration'] > 0.05:  # the duration of each shot should more than 50 ms
            valid_shots.append(shot)
    # plot it
    plt.figure()
    duration_info['duration'].plot.hist()
    plt.xlabel('duration time/s')
    plt.tight_layout()
    # plt.savefig('..//file_repo//info//ip.png')

    # %%
    # scan valid signal, create a dataframe to get all data by sample instead of shot
    tag_list = ["ip", "bt", "vl", "dx", "dy",
                "polaris_den_v01", "polaris_den_v09", "polaris_den_v17",
                'P_in', 'P_rad', 'ip_error', 'n=1 amplitude', 'ne0', 'ne_nG',
                'qa_proxy', 'radiation_proxy', 'rotating_mode_proxy',

                "sxr_cb_020", "sxr_cb_021", "sxr_cb_022", "sxr_cb_023",
                "sxr_cb_024", "sxr_cb_025", "sxr_cb_026", "sxr_cb_027", "sxr_cb_028",
                "sxr_cb_032",
                "sxr_cb_036", "sxr_cb_037", "sxr_cb_038", "sxr_cb_039", "sxr_cb_040",
                "sxr_cb_041", "sxr_cb_042", "sxr_cb_043", "sxr_cb_044",

                "sxr_cc_036", "sxr_cc_037", "sxr_cc_038", "sxr_cc_039",
                "sxr_cc_040", "sxr_cc_041", "sxr_cc_042", "sxr_cc_043", "sxr_cc_044",
                "sxr_cc_048",
                "sxr_cc_052", "sxr_cc_053", "sxr_cc_054", "sxr_cc_055", "sxr_cc_056",
                "sxr_cc_057", "sxr_cc_058", "sxr_cc_059", "sxr_cc_060",

                'AXUV_CA_02', 'AXUV_CA_06', 'AXUV_CA_10', 'AXUV_CA_14', 'AXUV_CB_18', 'AXUV_CB_22',
                'AXUV_CB_26', 'AXUV_CB_30', 'AXUV_CE_66', 'AXUV_CE_70', 'AXUV_CE_74', 'AXUV_CE_78',
                'AXUV_CF_82', 'AXUV_CF_86', 'AXUV_CF_90', 'AXUV_CF_94']  # target tags
    unvalid_shot = []
    shots_data = pd.DataFrame(columns=tag_list)  # empty dataframe
    for shot in valid_shots:
        shot_tag_list = source_file_repo.get_tag_list(shot)  # all tags in shot file
        if all(tag in shot_tag_list for tag in tag_list):
            data_raw = source_file_repo.read_data(shot, tag_list)
            # convert the dict to df, concatenate them to one df
            shots_data = pd.concat([shots_data, pd.DataFrame(data_raw)], ignore_index=True)
        else:
            unvalid_shot.append(shot)
    print(shots_data.shape)
    # the shots should be saved, both unvalid & the final used shots
    used_shots = list(filter(lambda x: x not in unvalid_shot, valid_shots))
    np.save('..//file_repo//info//std_info//used_shots.npy', used_shots)
    np.save('..//file_repo//info//std_info//unvalid_shots.npy', unvalid_shot)

    # %%
    # calculate the std, mean and so on
    result_df = shots_data.describe()
    result_df.to_csv('..//file_repo//info//std_info//result_statistics.csv', index=True)

    #%%
    # plot the hist of each signal
    for tag in tag_list:
        plt.figure()
        shots_data[tag].plot.hist()
        plt.xlabel('{}'.format(tag))
        plt.tight_layout()
        plt.savefig('..//file_repo//info//std_info//{}.png'.format(tag))
