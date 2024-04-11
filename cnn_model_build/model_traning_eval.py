# %%
# this examlpe shows how to build a ML mode to predict disruption and
# evaluate its performance using jddb
# this depands on the output FileRepo of basic_data_processing.py

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


def get_shot_result(y_red, threshold_sample, start_time):
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
    #%%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix//'
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))

    #%%
    # load model
    model = tf.keras.models.load_model('./best_model_all_mix')  # the model should be mypool one
    test_shot_list = test_file_repo.get_all_shots()
    # X = nn_data_build(56609, test_file_repo)
    # # get sample result from LightGBM
    # y_pred = model.predict(X)

    # create an empty result object
    test_result = Result(r'.\_temp_test\test_result.csv')
    sample_result = dict()

    # generate predictions for each shot
    shots_pred_disrurption = []  # shot predict result
    shots_pred_disruption_time = []  # shot predict time
    for shot in test_shot_list:
        X = nn_data_build(shot, test_file_repo)
        # get sample result from model
        y_pred = model.predict(X)
        sample_result.setdefault(shot, []).append(
            y_pred)  # save sample results to a dict

        # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        time_dict = test_file_repo.read_labels(shot, ['StartTime'])
        pred_disruption, predicted_disruption_time = get_shot_result(
            y_pred, .5, 0.182)  # get shot result by a threshold
        shots_pred_disrurption.append(pred_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)

    #%%
    # add predictions for each shot to the result object
    test_result.add(test_shot_list, shots_pred_disrurption,
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

    # # %% scan the threshold for shot prediction to get
    # # many results, and add them to a report
    # # simply change different disruptivity triggering level and logic, get many result.
    # test_report = Report('..//_temp_test//report.csv')
    # thresholds = np.linspace(0, 1, 50)
    # for threshold in thresholds:
    #     shots_pred_disrurption = []
    #     shots_pred_disruption_time = []
    #     for shot in test_shot_list:
    #         y_pred = sample_result[shot][0]
    #         predicted_disruption, predicted_disruption_time = get_shot_result(
    #             y_pred, threshold, 2.5)
    #         shots_pred_disrurption.append(predicted_disruption)
    #         shots_pred_disruption_time.append(predicted_disruption_time)
    #     # i dont save so the file never get created
    #     temp_test_result = Result('../_temp_test/temp_result.csv')
    #     temp_test_result.lucky_guess_threshold = .8
    #     temp_test_result.tardy_alarm_threshold = .001
    #     temp_test_result.add(test_shot_list, shots_pred_disrurption,
    #                          shots_pred_disruption_time)
    #     temp_test_result.get_all_truth_from_file_repo(test_file_repo)
    #
    #     # add result to the report
    #     test_report.add(temp_test_result, "thr=" + str(threshold))
    #     test_report.save()
    # # plot all metrics with roc
    # test_report.plot_roc('../_temp_test/')
