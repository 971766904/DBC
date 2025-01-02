#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/17 9:27
# @Author  : zhongyu
# @Site    : 
# @File    : roc_compare.py

'''
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

if __name__ == '__main__':
    # %%
    # load file_repo
    performance_data_dir = '..//..//file_repo//data_file//processed_data_cnn//fuwuqi'
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//dbc_shots_info.csv')

    # %%
    # plot roc curve of different dbc case
    dbcs = [62198, 61793, 61323, 38121, 30719, 23978]
    shots = [910, 718, 625, 478]
    shots_dbc = [1,2,3,4,5,6]
    myfont = fm.FontProperties(family='Times New Roman', size=16, weight='bold')
    font = {'family': 'Times New Roman', 'size': 16, 'weight': 'black'}
    plt.rc('font', **font)
    plt.figure()

    for case in shots:
        print('case:', case)
        case_info = pd.read_csv(os.path.join(performance_data_dir, 'compare_2//report_{}.csv'.format(case)))
        tpr = case_info['tpr']
        fpr = case_info['fpr']
        plt.plot(fpr, tpr, label='shots={}'.format(case))
    # add the font to the plot, include the title, x-axis and y-axis

    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('ROC curve of different shots case')
    plt.show()
