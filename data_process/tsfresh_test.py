#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/26 10:40
# @Author  : zhongyu
# @Site    : 
# @File    : tsfresh_test.py

'''
#%%
import numpy as np
from tsfresh import extract_features, select_features,feature_selection
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
import pandas as pd
from jddb.file_repo import FileRepo

if __name__ == '__main__':
    #%%
    #
    train_file_repo = FileRepo('..//..//file_repo//data_file//processed_data_1k_5k_label//$shot_2$00//')

    #%%
    #
    shots_list = train_file_repo.get_all_shots()
    tag_list = train_file_repo.get_tag_list(shots_list[0])
    data_raw = train_file_repo.read_data(shots_list[0], tag_list)
    n_sample = len(data_raw['ip'])
    data_raw['time'] = np.arange(n_sample)
    data_raw['shot'] = [shots_list[0]]*n_sample
    df_data = pd.DataFrame(data_raw)

    #%%
    # #
    # extracted_features = extract_features(df_data.drop(columns=['alarm_tag']),
    #                                       default_fc_parameters=EfficientFCParameters(),
    #                                       column_id='shot',
    #                                       column_sort='time',
    #                                       impute_function=impute)

    selected_features = feature_selection.significance_tests.target_binary_feature_real_test(
        df_data.drop(columns=['alarm_tag']), df_data['alarm_tag'], 'mann')



