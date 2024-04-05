#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/10/4 10:42
# @Author  : zhongyu
# @Site    : 
# @File    : train_test_model_tool.py

'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from jddb.performance import Result
from jddb.performance import Report
from jddb.file_repo import FileRepo


def matrix_build(shot_list, file_repo, tags,label):
    """
    get x and y from file_repo by shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = np.empty([0, len(tags)])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, tags)
        y_data = file_repo.read_data(shot, [label])
        x_data.pop(label, None)
        res = np.array(list(x_data.values())).T
        res_y = np.array(list(y_data.values())).T.flatten()
        x_set = np.append(x_set, res, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set

def matrix_build_stack(shot_list, file_repo, tags,label):
    """
    get x and y from file_repo by shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = np.empty([0, 25])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, tags)
        y_data = file_repo.read_data(shot, [label])
        x_data.pop(label, None)
        list_array = list(x_data.values())
        res = np.concatenate((list_array[0],list_array[1]),axis=1)
        res_y = np.array(list(y_data.values())).T.flatten()
        x_set = np.append(x_set, res, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set


def short_dis_matrix_build(shot_list, file_repo, tags):
    """
    get x and y from file_repo by shots and tags,this function remove the undisruption part of a disruption shot
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = np.empty([0, len(tags)])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, tags)
        y_data = file_repo.read_data(shot, ['alarm_tag'])
        dis_label = file_repo.read_labels(shot, ['IsDisrupt'])
        x_data.pop('alarm_tag', None)
        res = np.array(list(x_data.values())).T
        res_y = np.array(list(y_data.values())).T.flatten()
        if dis_label['IsDisrupt'] == 1:
            indices = np.where(res_y == 1)
            x_set = np.append(x_set, res[indices], axis=0)
            y_set = np.append(y_set, res_y[indices], axis=0)
        else:
            x_set = np.append(x_set, res, axis=0)
            y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set


def get_shot_result(y_pred, threshold_sample, duration, start_time):
    """
    get shot result by a threshold
    Args:
        y_red: sample result from model
        threshold_sample: disruptive predict level

    Returns: shot predict result and predict time

    """
    binary_result = 1 * (y_pred >= threshold_sample)
    for k in range(len(binary_result) - 2):
        if np.sum(binary_result[k:k + duration]) == duration:
            predicted_dis_time = (k + duration - 1) / 1000 + start_time
            predicted_dis = 1
            break
        else:
            predicted_dis_time = -1
            predicted_dis = 0
    return predicted_dis, predicted_dis_time
