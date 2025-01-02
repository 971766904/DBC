#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/8 15:39
# @Author  : zhongyu
# @Site    : 
# @File    : sample_eval.py

'''
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import h5py
from cnn import CNN
import pandas as pd
import os
from jddb.performance import Result
from jddb.performance import Report
from jddb.file_repo import FileRepo
from util.parse_tfrecord_pxuvbasic import parse_tfrecord


def matrix_build(shot_list, file_repo):
    """
    get x and y from file_repo by shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:


    Returns: matrix of x and y

    """
    x_set = np.empty([0, 32, 71])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, ['stacked_data'])
        y_data = file_repo.read_data(shot, ['label'])
        # convert the dict x_data with shape (1, none, 71,32)to ndarray res with shape (none,32, 71)
        array = np.squeeze(x_data['stacked_data'])

        # Transpose the array to get the shape (none, 32, 71)
        res = np.transpose(array, (0, 2, 1))

        res_y = np.squeeze(y_data['label'])
        x_set = np.append(x_set, res, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set


def nn_data_shot_build(shot_list, file_repo):
    data_array = np.empty([0, 32, 71])
    label_array = np.empty([0])

    for shot in shot_list:
        file_path = file_repo.get_file(shot)
        with h5py.File(file_path, 'r') as f:
            data = f.get('/data/stacked_data')[()]
            labels = f.get('/data/label')[()]  # get labels
            data = {"input_1": np.transpose(data, (0, 2, 1)),
                    "labels": labels}  # reshaping from (None, 71, 32) to (None, 32, 71)

        data_array = np.append(data_array, np.squeeze(data["input_1"]), axis=0)
        label_array = np.append(label_array, np.squeeze(data["labels"]), axis=0)

    return data_array, label_array


if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn'
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))

    # %%
    # get test data
    test_shot_list = test_file_repo.get_all_shots()
    X_test, y_test = matrix_build(test_shot_list, test_file_repo)

    # %%
    # load model
    X = {"input_1": X_test}
    model = tf.keras.models.load_model('./best_model_all_mix_1_tcn')  # the model should be mypool one
    y_pred = model.predict(X)

    # %%
    # assess the y_pred with y_test using confusion matrix,
    # the y_pred is the probability of the sample being disruptive, the y_test is the true label {0,1}
    y_pred = np.squeeze(y_pred)
    y_pred = 1 * (y_pred >= 0.5)
    y_test = np.squeeze(y_test)
    cm = tf.math.confusion_matrix(y_test, y_pred)
    print(cm)
    # %%
    # evaluate the result with auc
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_test, y_pred)
    print(f"auc is {auc}")

    # explain the confusion matrix of tf.math.confusion_matrix
    # [[TN FP]
    #  [FN TP]]

    # %%
    # calculate that how long do it take when nn_data_build is called
    import time

    t_start = time.time()
    data_list, label_list = nn_data_shot_build(test_shot_list, test_file_repo)
    t_end = time.time()
    print(f"nn_data_build takes {t_end - t_start} seconds")
