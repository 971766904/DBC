#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf-8 -*-
'''
# @Time    : 2023/10/7 14:37
# @Author  : zhongyu
# @Site    : 
# @File    : parse_tfrecord_pxuvbasic.py

'''
import tensorflow as tf


def parse_tfrecord(serialized_example):
    feature_description = {
        "input1": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
    }

    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    parsed_conv_input = tf.io.parse_tensor(parsed_example['input1'], tf.float16)
    parsed_disruptive = tf.io.parse_tensor(parsed_example['label'], tf.int8)

    parsed_conv_input = tf.reshape(parsed_conv_input, [32, 71])
    parsed_disruptive = tf.reshape(parsed_disruptive, ())

    input_dict = {
        "input_1": parsed_conv_input,
    }
    output_dict = parsed_disruptive
    print(input_dict)

    return input_dict, output_dict

if __name__ == '__main__':
    tfrecord_files = [f'../../../file_repo/data_file//processed_data_cnn//tfrecord/train/batch_{i}.tfrecord' for i in range(1, 5)]
    # tfrecord_files = './tfrecords'
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    print(dataset)
    for r1 in dataset.take(10):
        print(repr(r1))
    dataset = dataset.map(parse_tfrecord)
    for record in dataset.take(5):
        print(repr(record))
    # print(dataset.take(1).element_spec)
