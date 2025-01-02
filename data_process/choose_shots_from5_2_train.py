#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/12/25 0:54
# @Author  : zhongyu
# @Site    : 
# @File    : choose_shots_from5_2_train.py

'''
import pandas as pd
import numpy as np
from jddb.file_repo import FileRepo
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def dataset_shots_split(shots_1,shots_info):
    shots_is_disrupt = shots_info[shots_info['shot'].isin(shots_1)]['IsDisrupt']
    train_shots, test_shots, _, _ = \
        train_test_split(shots_1, shots_is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=shots_is_disrupt)
    return train_shots, test_shots



if __name__ == '__main__':
    # %%
    # load file
    dbc_shots_info = pd.read_csv('..//..//file_repo//info//std_info//power_dbc_shots_info.csv')
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    test_shots_1 = np.load('..//..//file_repo//info//split_dataset_info//test_shots_1.npy')
    train_shots_1 = np.load('..//..//file_repo//info//split_dataset_info//train_shots_1.npy')

    # %%
    # load eval test result csv
    test_eval_result = pd.read_csv('..//..//file_repo//ip_eval//case2_12_24_1//test_result.csv')

    #%%
    # get fn shots and fp shots
    fn_shots_result = test_eval_result[test_eval_result['false_negative']==1]
    fp_shots_result = test_eval_result[test_eval_result['false_positive']==1]
    # find out which category these shots belong to
    c5_fn_shots = fn_shots_result[fn_shots_result['shot_no'].isin(shots_5)]['shot_no']
    c5_fp_shots = fp_shots_result[fp_shots_result['shot_no'].isin(shots_5)]['shot_no']

    #%%
    # take part of category 5 shots from these shots to train set, the other category shots should
    # be reserved, then the final is test set
    fn_train_shots, fn_test_shots = train_test_split(c5_fn_shots, test_size=0.5, random_state=42)
    fp_train_shots, fp_test_shots = train_test_split(c5_fp_shots, test_size=0.5, random_state=42)
    train_shots_add5 = np.concatenate((fn_train_shots, fp_train_shots))       # the shots which are added to train set from c5
    train_shots_added = np.concatenate((train_shots_1,train_shots_add5))      # the train shots that contains added shots from c5 and former train shots
    # remove the train_shots_add5 from former test set filename:test_shots_1
    test_shots_split1 = np.setdiff1d(test_shots_1, train_shots_add5)          # the test shots being removed added shots

    #%%
    # save the train_shots as npy file
    np.save('..//..//file_repo//info//split_dataset_info//train_add5.npy',train_shots_add5)
    np.save('..//..//file_repo//info//split_dataset_info//test_shots_split1.npy', test_shots_split1)
    np.save('..//..//file_repo//info//split_dataset_info//train_shots_added.npy',train_shots_added)


    

# %%
