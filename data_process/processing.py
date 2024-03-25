#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/18 16:33
# @Author  : zhongyu
# @Site    : 
# @File    : processing.py

'''
#%%
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from util.basic_processor import  AlarmTag
from jddb.processor.basic_processors import ResamplingProcessor, TrimProcessor
import pandas as pd

if __name__ == '__main__':
    #%%
    # load filerepo
    common_signal = []
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    train_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_1k_5k_label//$shot_2$00//')
    source_shotset = ShotSet(source_file_repo)
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
                'AXUV_CF_82', 'AXUV_CF_86', 'AXUV_CF_90', 'AXUV_CF_94']

    #%%
    # add disruption labels for each time point as a signal called alarm_tag
    processed_shotset = source_shotset.remove_signal(tags=tag_list, keep=True,
                                                     save_repo=train_file_repo)
    processed_shotset = processed_shotset.process(
        processor=AlarmTag(
            lead_time=0.03, disruption_label="IsDisrupt", downtime_label="DownTime"),
        input_tags=["ip"],
        output_tags=["alarm_tag"],
        save_repo=train_file_repo,
        processes=10)

