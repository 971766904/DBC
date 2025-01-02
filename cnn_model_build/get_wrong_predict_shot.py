#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/13 16:59
# @Author  : zhongyu
# @Site    : 
# @File    : get_wrong_predict_shot.py

'''
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


def fig_shot_predict(y_pred, file_repo, shot, predicted_disruption):
    true_disruption = 0 if file_repo.read_labels(shot)["IsDisrupt"] == False else 1

    if not (true_disruption == predicted_disruption):
        t_start = file_repo.read_labels(shot, ['StartTime'])
        t = t_start['StartTime'] + np.arange(y_pred.shape[0]) * 0.001
        plt.figure()
        ax1 = plt.subplot(111)
        # y axis limit in[0,1]
        ax1.set_ylim(0, 1)
        ax1.plot(t, y_pred, 'r')
        # ax2 = ax1.twinx()
        # ax2.plot(t, X[:, 21])
        plt.title('true:{}'.format(true_disruption))
        plt.savefig('./_temp_fig/{}.png'.format(shot))
        plt.close()
        return shot
    return None


# %% define function to build model specific data

def nn_data_build(shot_no, file_repo):
    shot_no = int(shot_no)
    file_path = file_repo.get_file(shot_no)
    test_data = h5py.File(file_path, 'r')
    X = test_data.get('/data/stacked_data')[()]
    X = {"input_1": np.transpose(X, (0, 2, 1))}  # reshaping from (None, 71, 32) to (None, 32, 71)
    return X


def simple_nn_data_build(shot_no, file_repo):
    shot_no = int(shot_no)
    file_path = file_repo.get_file(shot_no)
    test_data = h5py.File(file_path, 'r')
    X = test_data.get('/data/basic')[()]
    return X


# inference on shot


def get_shot_result(y_pred, threshold_sample, start_time):
    """
    get shot result by a threshold and compare to start time
    Args:
        y_red: sample result from model
        threshold_sample: disruptive predict level
        start_time: shot start time

    Returns:
        shot predict result:The prediction result for the shot
        predict time: The time when disruption prediction is made or -1 for no disruption shot

    """
    predicted_dis = 0
    predicted_dis_time = -1
    binary_result = 1 * (y_pred >= threshold_sample)
    for k in range(len(binary_result) - 2):
        if np.sum(binary_result[k:k + 3]) == 3:
            predicted_dis_time = (k + 2) / 1000 + start_time
            predicted_dis = 1
            break
        else:
            predicted_dis_time = -1
            predicted_dis = 0
    return predicted_dis, predicted_dis_time


# %% init FileRepo
if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix//'
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))
    wrong_shot_path = './wrong_predict_shots.npy'

    # %%
    # load model
    model = tf.keras.models.load_model('./best_model_all_mix')  # the model should be mypool one
    test_shot_list = test_file_repo.get_all_shots()

    # generate predictions for each shot
    shots_pred_disrurption = []  # shot predict result
    shots_pred_disruption_time = []  # shot predict time
    wrong_predict_shots = []
    for shot in test_shot_list:
        X = nn_data_build(shot, test_file_repo)
        # get sample result from model
        y_pred = model.predict(X)

        # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        time_dict = test_file_repo.read_labels(shot, ['StartTime'])
        pred_disruption, predicted_disruption_time = get_shot_result(
            y_pred, .5, 0.182)  # get shot result by a threshold
        predict = fig_shot_predict(y_pred, test_file_repo, shot, pred_disruption)
        if predict:
            wrong_predict_shots.append(predict)
        shots_pred_disrurption.append(pred_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)
    # save wrong predict shots
    np.save(wrong_shot_path, wrong_predict_shots)
