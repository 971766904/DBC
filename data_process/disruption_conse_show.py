#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/29 22:51
# @Author  : zhongyu
# @Site    : 
# @File    : disruption_conse_show.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import pandas as pd
import os
from jddb.file_repo import FileRepo
from multipledispatch.dispatcher import source

from util.fig_plot import Plotsignal,Plot2d


if __name__ == '__main__':
    #%%
    # load shots file and file repo
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    dbc_data_dir = '..//..//file_repo//data_file//processed_lgbm'
    source_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_1k_5k_label//$shot_2$00//')  # the raw signal file repo
    # get tags
    shots_all= source_file_repo.get_all_shots()
    tags_all = source_file_repo.get_tag_list(shots_all[0])
    array_tags = ['sxr_cc_036', 'sxr_cc_037', 'sxr_cc_038', 'sxr_cc_039', 'sxr_cc_040', 'sxr_cc_041', 'sxr_cc_042',
                  'sxr_cc_043', 'sxr_cc_044', 'sxr_cc_048', 'sxr_cc_052', 'sxr_cc_053', 'sxr_cc_054', 'sxr_cc_055',
                  'sxr_cc_056', 'sxr_cc_057', 'sxr_cc_058', 'sxr_cc_059', 'sxr_cc_060']
    tags_plot = ['P_in', 'P_rad', 'ip', 'n=1 amplitude', 'ne0', 'ne_nG','vl','radiation_proxy','qa_proxy',
                 'rotating_mode_proxy']
    # get c5 disruption shots
    dis_shots = []
    undis_shots = []
    for shot in shots_5:
        disrup_label = source_file_repo.read_labels(shot, ['IsDisrupt'])['IsDisrupt']
        if disrup_label:
            dis_shots.append(shot)
        else:
            undis_shots.append(shot)


    # #%%
    # # show the shot signal fig
    #
    # shots_plot = Plotsignal(file_repo=source_file_repo,tag_list=tags_plot,shot=shots_5[0])
    # shots_plot.signal_plot()
    #
    # #%%
    # # array plot analysis
    #
    # array_fig = Plot2d(array_tags,shots_5[6],source_file_repo)
    # array_fig.array_plot()

    #%%
    # plot all fig
    for shot in undis_shots:
        shots_plot = Plotsignal(file_repo=source_file_repo, tag_list=tags_plot, shot=shot)
        shots_plot.signal_plot()

        array_fig = Plot2d(array_tags, shot, source_file_repo)
        array_fig.array_plot()