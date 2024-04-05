#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/11/13 12:18
# @Author  : zhongyu
# @Site    : 
# @File    : train_cnn.py

'''
# %%
import tensorflow as tf
from util.parse_tfrecord_pxuvbasic import parse_tfrecord
from lstm import CNN_LSTM,ConvLSTM1d
from cnn import CNN
from keras.callbacks import EarlyStopping
from tensorflow import keras

# %%

tf.random.set_seed(42)

if __name__ == '__main__':
    # %%
    # load tfrecord
    BATCH_SIZE = 1024 * 2
    train_files = tf.data.Dataset.list_files('..//..//file_repo//data_file//processed_data_cnn//tfrecord/train/*.tfrecord')
    val_files = tf.data.Dataset.list_files('..//..//file_repo//data_file//processed_data_cnn//tfrecord/val/*.tfrecord')

    train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(BATCH_SIZE).cache()
    # %%
    # training
    model = CNN()
    early_stopping = EarlyStopping(monitor='val_auc', patience=10, verbose=2)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.AUC(name='auc')])

    # history = model.fit(train, batch_size=20, epochs=10, callbacks=[early_stopping], validation_data=val)
    history = model.fit(train_ds, batch_size=BATCH_SIZE, epochs=5,
                        validation_data=val_ds)
    model.compile()
    model.summary()
    # val = val.batch(200)
    prediction = model.predict(val_ds)
    # %%
    # # save model
    # model.save('./et1_lstm_tfrecord')
