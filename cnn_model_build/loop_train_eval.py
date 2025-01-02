#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/15 10:26
# @Author  : zhongyu
# @Site    : 
# @File    : loop_train_eval.py

'''
from cnn import CNN, CNN1, TCNModel
from keras.callbacks import EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from jddb.performance import Result
from jddb.performance import Report
from jddb.file_repo import FileRepo
from util.parse_tfrecord_pxuvbasic import parse_tfrecord
from model_traning_eval import nn_data_build, get_shot_result
from sample_eval import nn_data_shot_build
from sklearn.metrics import roc_auc_score

tf.random.set_seed(42)


def train(num_round):
    # %%
    # load tfrecord
    BATCH_SIZE = 1024 * 2
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    training_shots = np.load(
        os.path.join(dbc_data_dir, 'info//dbc//dbc_training_shots_{}.npy'.format(num_round)))  # change file path
    valid_shots = np.load(
        os.path.join(dbc_data_dir, 'info//dbc//dbc_valid_shots_{}.npy'.format(num_round)))  # change file path
    model_path = './dbc_shots/model/best_model_{}'.format(num_round)                       # change file path
    training_path = [os.path.join(dbc_data_dir, f'dbc_shots/{shot}.tfrecord') for shot in training_shots]
    val_path = [os.path.join(dbc_data_dir, f'dbc_shots/{shot}.tfrecord') for shot in valid_shots]
    train_files = tf.data.Dataset.list_files(training_path)

    val_files = tf.data.Dataset.list_files(val_path)

    #
    train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    # %%
    # training
    model = TCNModel()  # change model name

    from keras.callbacks import ModelCheckpoint

    # Define the early stopping callback
    early_stopping = EarlyStopping(monitor='val_auc', patience=20, verbose=2)

    # Define the model checkpoint callback
    # This will save the entire model to the directory specified in `filepath`
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_auc',
                                       verbose=1, save_best_only=True, mode='max', save_weights_only=False)

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(name='auc')])

    # Fit the model
    history = model.fit(train_ds, batch_size=BATCH_SIZE, epochs=100,
                        validation_data=val_ds, callbacks=[early_stopping, model_checkpoint])


def eval(num_round):
    # %%
    # load file repo
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    eval_dir = 'dbc'
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))

    # %%
    # load model
    model = tf.keras.models.load_model(
        os.path.join(eval_dir, 'model/best_model_{}'.format(num_round)))  # the model should be change
    test_shot_list = test_file_repo.get_all_shots()
    # X = nn_data_build(56609, test_file_repo)
    # # get sample result from LightGBM
    # y_pred = model.predict(X)

    # create an empty result object
    test_result = Result(os.path.join(eval_dir, 'result/_temp_test_{}/test_result.csv'.format(num_round)))  # change file path
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
        time_dict = test_file_repo.read_attributes(shot, 'label', ['StartTime'])
        pred_disruption, predicted_disruption_time = get_shot_result(
            y_pred, .5, time_dict['StartTime'])  # get shot result by a threshold
        # fig_shot_predict(y_pred, test_file_repo, shot, pred_disruption)
        shots_pred_disrurption.append(pred_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)

    # %%
    # add predictions for each shot to the result object
    print(sum(shots_pred_disrurption))
    test_result.add(test_shot_list, shots_pred_disrurption,
                    shots_pred_disruption_time)
    # get true disruption label and time
    test_result.get_all_truth_from_file_repo(
        test_file_repo)

    test_result.lucky_guess_threshold = 2
    test_result.tardy_alarm_threshold = .005
    test_result.calc_metrics()
    test_result.save()

    # %% scan the threshold for shot prediction to get
    # many results, and add them to a report
    # simply change different disruptivity triggering level and logic, get many result.
    test_report = Report(os.path.join(eval_dir, 'result/_temp_test_{}//report.csv'.format(num_round)))  # change file path
    thresholds = np.linspace(0, 1, 50)
    for threshold in thresholds:
        shots_pred_disrurption = []
        shots_pred_disruption_time = []
        for shot in test_shot_list:
            y_pred = sample_result[shot][0]
            time_dict = test_file_repo.read_attributes(shot, 'label', ['StartTime'])
            predicted_disruption, predicted_disruption_time = get_shot_result(
                y_pred, threshold, time_dict['StartTime'])
            shots_pred_disrurption.append(predicted_disruption)
            shots_pred_disruption_time.append(predicted_disruption_time)
        # i dont save so the file never get created
        temp_test_result = Result(
            os.path.join(eval_dir, 'result/_temp_test_{}/temp_result.csv'.format(num_round)))  # change file path
        temp_test_result.lucky_guess_threshold = 2
        temp_test_result.tardy_alarm_threshold = .001
        temp_test_result.add(test_shot_list, shots_pred_disrurption,
                             shots_pred_disruption_time)
        temp_test_result.get_all_truth_from_file_repo(test_file_repo)

        # add result to the report
        test_report.add(temp_test_result, "thr=" + str(threshold))
        test_report.save()
    # plot all metrics with roc
    test_report.plot_roc(os.path.join(eval_dir, 'result/_temp_test_{}/'.format(num_round)))  # change file path

def sample_eval(num_round):
    # %%
    # load file repo
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    eval_dir = 'dbc'
    # load csv file
    info_path = os.path.join(dbc_data_dir, 'info//std_info//dbc_shots_info.csv')
    dbc_shots_info = pd.read_csv(info_path)
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'label_test//$shot_2$00//'))

    # %%
    # load model
    model = tf.keras.models.load_model(
        os.path.join(eval_dir, 'model/best_model_{}'.format(num_round)))  # the model should be change
    test_shot_list = test_file_repo.get_all_shots()
    X_test, y_test = nn_data_shot_build(test_shot_list, test_file_repo)

    X = {"input_1": X_test}
    y_pred = model.predict(X)

    auc = roc_auc_score(y_test, y_pred)
    print(f"auc is {auc}")
    # write auc to dbc_shots_info
    dbc_shots_info.loc[num_round-1,'sample_auc'] = auc

if __name__ == '__main__':
    for i in range(10):
        train(i+1)
        eval(i+1)
