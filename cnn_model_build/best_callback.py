#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/9 16:32
# @Author  : zhongyu
# @Site    : 
# @File    : best_callback.py

'''
# %%
import tensorflow as tf
from util.parse_tfrecord_pxuvbasic import parse_tfrecord
from lstm import CNN_LSTM, ConvLSTM1d
from cnn import CNN, CNN1
from keras.callbacks import EarlyStopping
from tensorflow import keras
import matplotlib.pyplot as plt
import os


def training_fig(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()


# %%

tf.random.set_seed(42)

if __name__ == '__main__':
    # %%
    # load tfrecord
    BATCH_SIZE = 1024 * 2
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix'
    model_path = './best_model_all_mix'
    train_files = tf.data.Dataset.list_files(
        os.path.join(dbc_data_dir, 'tfrecord/train/*.tfrecord'))
    val_files = tf.data.Dataset.list_files(
        os.path.join(dbc_data_dir, 'tfrecord/val/*.tfrecord'))

    #
    train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    # %%
    # training
    model = CNN1()

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
    training_fig(history)

    # %%
    # assess the model
    # Predict the train_inputs using the model
    import numpy as np

    model = tf.keras.models.load_model(model_path)

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
