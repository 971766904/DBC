#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/11/27 13:44
# @Author  : zhongyu
# @Site    : 
# @File    : model_eval.py

'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


if __name__ == '__main__':
    # %%
    # init FileRepo
    dbc_data_dir = '..//..//file_repo//data_file//processed_lgbm'
    test_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))
    test_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_test_all//$shot_2$00//'))

    #load test shots
    shots_5 = np.load('..//..//file_repo//info//ip_info//ip_5.npy')
    shots_4 = np.load('..//..//file_repo//info//ip_info//ip_4.npy')
    shots_random_test = np.load('..//..//file_repo//info//ip_info//randm_test_shots.npy')
    # eliminate the shots in test_shots.npy from shots_5
    shots_test_wrong = np.load('..//..//file_repo//info//ip_info//chosen_shots_undis.npy')
    shots_5 = np.setdiff1d(shots_5, shots_test_wrong)
    test_shots = shots_5
    test_shots = np.load('..//..//file_repo//info//split_dataset_info//shots5_reserve.npy')

    #load model
    model = lgb.Booster(model_file='model//lgbm_model_added_1.txt')

    # init result
    test_result = Result(r'_temp_test\test_result.csv')
    sample_result = dict()
    shots_pred_disrurption = []  # shot predict result
    shots_true_disruption = []  # shot true disruption label
    shots_pred_disruption_time = []  # shot predict time

    #%%
    #get result from model inference
    for shot in test_shots:
        X,_ = matrix_build([shot], test_file_repo, ['stacked_data'],71,['label'])
        # get sample result from model
        y_pred = model.predict(X)
        sample_result.setdefault(shot, []).append(
            y_pred)  # save sample results to a dict

        # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        time_dict = test_file_repo.read_attributes(shot, 'label', ['StartTime'])
        pred_disruption, predicted_disruption_time = get_shot_result(
            y_pred, .5, time_dict['StartTime'])  # get shot result by a threshold
        # fig_shot_predict(y_pred, test_file_repo, shot, pred_disruption)
        shots_pred_disrurption.append(pred_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)

        # %%
        # add predictions for each shot to the result object
    test_result.add(test_shots, shots_pred_disrurption,
                    shots_pred_disruption_time)
    # get true disruption label and time
    test_result.get_all_truth_from_file_repo(
        test_file_repo)

    test_result.lucky_guess_threshold = 2
    test_result.tardy_alarm_threshold = .005
    test_result.calc_metrics()
    test_result.save()
    print("precision = " + str(test_result.precision))
    print("tpr = " + str(test_result.tpr))

    # %% plot some of the result: confusion matrix, warning time histogram
    # and accumulate warning time.
    sns.heatmap(test_result.confusion_matrix, annot=True, cmap="Blues", fmt='.0f')
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    # plt.savefig(os.path.join('..//_temp_test//', 'Confusion Matrix.png'), dpi=300)
    plt.show()

    test_result.plot_warning_time_histogram(
        [-1, .002, .01, .05, .1, .3], './/_temp_test//')
    test_result.plot_accumulate_warning_time('.//_temp_test//')

    # %% scan the threshold for shot prediction to get
    # many results, and add them to a report
    # simply change different disruptivity triggering level and logic, get many result.
    test_report = Report('.//_temp_test//report.csv')
    thresholds = np.linspace(0, 1, 50)
    for threshold in thresholds:
        shots_pred_disrurption = []
        shots_pred_disruption_time = []
        for shot in test_shots:
            y_pred = sample_result[shot][0]
            time_dict = test_file_repo.read_attributes(shot, 'label', ['StartTime'])
            predicted_disruption, predicted_disruption_time = get_shot_result(
                y_pred, threshold, time_dict['StartTime'])
            shots_pred_disrurption.append(predicted_disruption)
            shots_pred_disruption_time.append(predicted_disruption_time)
        # i dont save so the file never get created
        temp_test_result = Result('./_temp_test/temp_result.csv')
        temp_test_result.lucky_guess_threshold = 2
        temp_test_result.tardy_alarm_threshold = .001
        temp_test_result.add(test_shots, shots_pred_disrurption,
                             shots_pred_disruption_time)
        temp_test_result.get_all_truth_from_file_repo(test_file_repo)

        # add result to the report
        test_report.add(temp_test_result, "thr=" + str(threshold))
        test_report.save()
    # plot all metrics with roc
    test_report.plot_roc('./_temp_test/')