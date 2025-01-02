#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/19 10:18
# @Author  : zhongyu
# @Site    : 
# @File    : fold_cross_validation.py

'''
import tensorflow as tf
from util.parse_tfrecord_pxuvbasic import parse_tfrecord
from lstm import CNN_LSTM, ConvLSTM1d
from cnn import CNN, CNN1, TCNModel
from keras.callbacks import EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import numpy as np
from best_callback import training_fig
from sklearn.model_selection import KFold

tf.random.set_seed(42)

if __name__ == '__main__':
    # %%
    # load tfrecord
    BATCH_SIZE = 1024 * 2
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    model_path = './fold_cross_validation/model/best_model'
    # load training val shots
    train_val = np.load(os.path.join(dbc_data_dir, 'info//train+val//training_shots_{}.npy'))  # change file path
    # Define the KFold cross-validator
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    # Loop over the splits
    for train_indices, val_indices in kfold.split(train_val):
        # Create datasets for the training and validation indices
        training_shots = train_val[train_indices]
        valid_shots = train_val[val_indices]
        # load tfrecord
        training_path = [os.path.join(dbc_data_dir, f'tfrecord_train+val/{shot}.tfrecord') for shot in training_shots]
        val_path = [os.path.join(dbc_data_dir, f'tfrecord_train+val/{shot}.tfrecord') for shot in valid_shots]
        train_files = tf.data.Dataset.list_files(training_path)

        val_files = tf.data.Dataset.list_files(val_path)

        #
        train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
            map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
            map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

        train_ds = train_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
        val_ds = val_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
        # training
        model = CNN1()

        from keras.callbacks import ModelCheckpoint

        # Define the early stopping callback
        early_stopping = EarlyStopping(monitor='val_auc', patience=20, verbose=2)

        # Define the model checkpoint callback
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_auc',
                                           verbose=1, save_best_only=True, mode='max', save_weights_only=False)

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.AUC(name='auc')])

        # Fit the model
        history = model.fit(train_ds, epochs=100,
                            validation_data=val_ds, callbacks=[early_stopping, model_checkpoint])
        training_fig(history)

        train_ds_s = val_ds.map(lambda x, y: (x, y))
        train_inputs = train_ds_s.map(lambda x, y: x)
        train_labels = train_ds_s.map(lambda x, y: y)
        y_pred = model.predict(train_inputs)

        # Convert the predicted probabilities to class labels
        y_pred = np.squeeze(y_pred)
        y_pred_labels = 1 * (y_pred >= 0.5)

        # Convert the train_labels to numpy array for confusion matrix calculation
        train_labels_np = np.concatenate([y for y in train_labels], axis=0)

        # Generate the confusion matrix
        cm = tf.math.confusion_matrix(train_labels_np, y_pred_labels)

        # Print the confusion matrix
        print(cm)
