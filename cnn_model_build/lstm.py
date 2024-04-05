#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/11/13 10:08
# @Author  : zhongyu
# @Site    : 
# @File    : lstm.py

'''
import tensorflow as tf
import numpy as np
from keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense,TimeDistributed


class CNN_LSTM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )

        self.pool1 = TimeDistributed(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
        self.flatten = TimeDistributed(tf.keras.layers.Flatten())

        self.lstm = tf.keras.layers.LSTM(128)

        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense2 = layers.Dense(64, activation='relu')

        self.drop = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        conv_inputs = inputs["input_1"]
        basic_inputs = inputs["input_2"]
        c0 = layers.Reshape(target_shape=(conv_inputs.shape[1],64, 1))(conv_inputs)
        print(c0.shape)
        # l0 = layers.Reshape(target_shape=(basic_inputs.shape[1]))(basic_inputs)
        c = self.conv1(c0)
        print(c.shape)
        c = self.pool1(c)
        print(c.shape)
        c = self.flatten(c)
        print(c.shape)
        x = layers.Concatenate()([c, basic_inputs])
        x = self.lstm(x)
        x = self.dense1(x)  # [batch_size, 1024]

        x = self.drop(x)
        output = self.dense3(x)  # [batch_size, 1]

        return output

class ConvLSTM1d(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.convlstm = tf.keras.layers.ConvLSTM1D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation='tanh',  # 激活函数
        )

        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.flatten = tf.keras.layers.Flatten()

        self.lstm = tf.keras.layers.LSTM(128)

        self.dense1 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)
        self.dense2 = layers.Dense(64, activation='relu')

        self.drop = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        conv_inputs = inputs["input_1"]
        basic_inputs = inputs["input_2"]
        c0 = layers.Reshape(target_shape=(conv_inputs.shape[1],64, 1))(conv_inputs)
        print(c0.shape)
        # l0 = layers.Reshape(target_shape=(basic_inputs.shape[1]))(basic_inputs)
        c = self.convlstm(c0)
        print(c.shape)
        c = self.pool1(c)
        print(c.shape)
        c = self.flatten(c)
        print(c.shape)
        b = self.lstm(basic_inputs)
        x = layers.Concatenate()([c, b])

        x = self.dense1(x)  # [batch_size, 1024]

        x = self.drop(x)
        output = self.dense3(x)  # [batch_size, 1]

        return output