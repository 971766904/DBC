#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf-8 -*-
'''
# @Time    : 2024/4/9 10:15
# @Author  : zhongyu
# @Site    : 
# @File    : tfrecord_pase_test.py

'''
# %%
import tensorflow as tf
import os
import h5py
import numpy as np
import tensorflow as tf
from tfrecord_dataset import sample_generator, create_tfrecord_writer
from jddb.file_repo import FileRepo


def matrix_build(shot_list, file_repo):
    """
    get x and y from file_repo by shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:


    Returns: matrix of x and y

    """
    x_set = np.empty([0, 32, 71])
    y_set = np.empty([0])
    for shot in shot_list:
        shot = int(shot)
        x_data = file_repo.read_data(shot, ['stacked_data'])
        y_data = file_repo.read_data(shot, ['label'])
        # convert the dict x_data with shape (1, none, 71,32)to ndarray res with shape (none,32, 71)
        array = np.squeeze(x_data['stacked_data'])

        # Transpose the array to get the shape (none, 32, 71)
        res = np.transpose(array, (0, 2, 1))

        res_y = np.squeeze(y_data['label'])
        x_set = np.append(x_set, res, axis=0)
        y_set = np.append(y_set, res_y, axis=0)
    return x_set, y_set


def parse_tfrecord(serialized_example):
    feature_description = {
        "input1": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }

    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    parsed_conv_input = tf.io.parse_tensor(parsed_example['input1'], tf.float16)
    parsed_disruptive = tf.io.parse_tensor(parsed_example['label'], tf.int8)

    parsed_conv_input = tf.reshape(parsed_conv_input, [71, 32])
    parsed_disruptive = tf.reshape(parsed_disruptive, ())
    parsed_conv_input = tf.transpose(parsed_conv_input, [1, 0])

    input_dict = {
        "input_1": parsed_conv_input,
    }
    output_dict = parsed_disruptive
    print(input_dict)

    return input_dict, output_dict


def parse_tfrecord_1(example_proto):
    # Define the features in the TFRecord file
    feature_description = {
        'input1': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.Example proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the feature1 data
    parsed_example['input1'] = tf.io.decode_raw(parsed_example['input1'], tf.float32)
    parsed_example['label'] = tf.io.decode_raw(parsed_example['label'], tf.int8)

    input_dict = {
        "input_1":  parsed_example['input1'],
    }
    output_dict = parsed_example['label']

    return input_dict, output_dict

if __name__ == '__main__':
    # %%
    # save as tfrecord file
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn'
    file_dirs = [
        os.path.join(dbc_data_dir, 'test_h5')]
    tfrecord_dirs = [
        os.path.join(dbc_data_dir, 'test_tfrecord')]
    for file_dir, tfrecord_dir in zip(file_dirs, tfrecord_dirs):
        file_paths = tf.data.Dataset.list_files(str(file_dir + '/*/*.hdf5')).as_numpy_iterator()
        batch_size = 1024 * 8
        output_dir = tfrecord_dir
        sample_gen = sample_generator(file_paths)

        file_counter = 1
        sample_counter = 0
        writer = create_tfrecord_writer(file_counter, output_dir)
        for sample in sample_gen:
            feature_dict = dict()

            for key, value in sample.items():
                feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))
                feature_dict[key] = feature

            features = tf.train.Features(feature=feature_dict)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sample_counter += 1

            if sample_counter % batch_size == 0:
                writer.close()
                file_counter += 1
                writer = create_tfrecord_writer(file_counter)

    # %%
    # read the tfrecord file

    train_files = tf.data.Dataset.list_files(
        '..//..//file_repo//data_file//processed_data_cnn//test_tfrecord/*.tfrecord')

    train_ds = train_files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.AUTOTUNE). \
        map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    for element in train_ds.take(5):
        print(element)

    # %%
    # read the h5 file
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn'
    test_file_repo = FileRepo(
        os.path.join(dbc_data_dir, 'test_h5//$shot_2$00//'))
    test_shot_list = test_file_repo.get_all_shots()
    X_test, y_test = matrix_build(test_shot_list, test_file_repo)
