#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/3 14:13
# @Author  : zhongyu
# @Site    : 
# @File    : cnn.py

'''
import tensorflow as tf
import numpy as np
from keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Dense, TimeDistributed
from tcn import TCN


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=128,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.conv3 = tf.keras.layers.Conv1D(
            filters=256,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )

        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.pool2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.gapool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.drop = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        conv_inputs = inputs["input_1"]
        # c0 = layers.Reshape(target_shape=(conv_inputs.shape[1], 71, 1))(conv_inputs)
        # print(c0.shape)
        # l0 = layers.Reshape(target_shape=(basic_inputs.shape[1]))(basic_inputs)
        c = self.conv1(conv_inputs)
        print(c.shape)
        c = self.pool1(c)
        print(c.shape)
        c = self.conv2(c)
        print(c.shape)
        c = self.pool2(c)
        print(c.shape)
        c = self.conv3(c)
        print(c.shape)
        c = self.gapool(c)
        print(c.shape)

        x = self.dense1(c)  # [batch_size, 1024]

        x = self.drop(x)
        output = self.dense3(x)  # [batch_size, 1]

        return output


class CNN1(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64,  # 卷积层神经元（卷积核）数目
            kernel_size=3,  # 感受野大小
            padding='valid',  # padding策略（vaild 或 same）
            activation=tf.nn.relu,  # 激活函数
        )

        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        # self.pool1 = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling1D())
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.drop = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(units=1, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        conv_inputs = inputs["input_1"]
        # c0 = layers.Reshape(target_shape=(conv_inputs.shape[1], 71, 1))(conv_inputs)
        # print(c0.shape)
        # l0 = layers.Reshape(target_shape=(basic_inputs.shape[1]))(basic_inputs)
        c = self.conv1(conv_inputs)
        print(c.shape)

        c = self.pool1(c)
        c = self.flatten(c)
        print(c.shape)

        x = self.dense1(c)  # [batch_size, 1024]

        x = self.drop(x)
        output = self.dense3(x)  # [batch_size, 1]

        return output


class TCNModel(tf.keras.Model):
    """
    Temporal Convolutional Network (TCN) model for binary classification.
    """

    def __init__(self, nb_filters=64, kernel_size=3, dropout_rate=0.5):
        """
        Initialize the TCN model.

        Args:
            nb_filters (int): The number of filters to use in the TCN layer.
            kernel_size (int): The size of the kernel to use in the TCN layer.
            dropout_rate (float): The dropout rate.
        """
        super(TCNModel, self).__init__()
        self.tcn1 = TCN(nb_filters=nb_filters, kernel_size=kernel_size, return_sequences=False)
        self.dense1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.drop = tf.keras.layers.Dropout(dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Forward pass for the model.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        conv_inputs = inputs["input_1"]
        x = self.tcn1(conv_inputs)
        x = self.dense1(x)
        x = self.drop(x)
        output = self.dense2(x)
        return output
