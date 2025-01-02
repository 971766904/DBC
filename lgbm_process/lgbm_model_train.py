#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/26 16:10
# @Author  : zhongyu
# @Site    : 
# @File    : lgbm_model_train.py

'''
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import os


from jddb.performance import Result
from jddb.performance import Report

from jddb.file_repo import FileRepo

def matrix_build(shot_list, file_repo, tags,tag_length,label):
    """
    get x and y from file_repo with shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = np.empty([0, tag_length])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        try:
            x_data = file_repo.read_data(shot, tags)
            y_data = file_repo.read_data(shot, label)
        except:
            print(f'error shot: {shot}')
            continue
        _temp_value = list(x_data.values())
        _ndarray = np.array(_temp_value).reshape(-1, tag_length)
        res_y = np.array(list(y_data.values())).T.flatten()
        x_set = np.append(x_set, _ndarray, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set

if __name__ == '__main__':
    # %%
    # init FileRepo
    dbc_data_dir = '..//..//file_repo//data_file//processed_lgbm'
    file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train_all//$shot_2$00//'))



    # %%
    # shots list load
    training_shots = np.load('..//..//file_repo//info//split_dataset_info//train_shots_added.npy')


    # get the label of training shots  as is_disrupt
    is_disrupt = []
    for shot in training_shots:
        # if shot is not int, change it to int
        shot = int(shot)
        dis_label = file_repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])

    # split the training set to train and validation set
    train_shots, val_shots, _, _ = \
        train_test_split(training_shots, is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=is_disrupt)


    #matrix build
    X_train, y_train = matrix_build(train_shots, file_repo, ['stacked_data'],71,['label'])
    X_val, y_val = matrix_build(val_shots, file_repo, ['stacked_data'],71,['label'])
    # X_test, y_test = matrix_build(test_shots, test_file_repo, ['stacked_data'],71,['label'])
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val)

    # train model with lightgbm
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 9,
        'num_leaves': 70,
        'learning_rate': 0.1,
        'feature_fraction': 0.86,
        'bagging_fraction': 0.73,
        'bagging_freq': 0,
        'verbose': 0,
        'cat_smooth': 10,
        'max_bin': 255,
        'min_data_in_leaf': 165,
        'lambda_l1': 0.03,
        'lambda_l2': 2.78,
        'is_unbalance': True,
        'min_split_gain': 0.3

    }
    evals_result = {}  # to record eval results for plotting
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets={lgb_train, lgb_val},
                    evals_result=evals_result,
                    early_stopping_rounds=30)

    # # save model in model folder
    # gbm.save_model('model//lgbm_model.txt')