#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/2 17:08
# @Author  : zhongyu
# @Site    : 
# @File    : tfrecord_dataset.py

'''
import os
import h5py
import tensorflow as tf


def sample_generator(file_paths):
    while True:
        try:
            current_file = next(file_paths)
        except StopIteration:
            break
        try:
            with h5py.File(current_file, 'r') as h5_file:
                # Iterate over samples in the current file
                conv_input = h5_file.get('/data/stacked_data')[()].astype('float16')
                disruptive = h5_file.get('/data/label')[()].astype('int8')
                for idx in range(disruptive.shape[0]):
                    sample_dict = {
                        "input1": conv_input[idx],
                        "label": disruptive[idx],
                    }
                    yield sample_dict  # Yield the sample
        except Exception as e:
            print(f'current file: {current_file}\nException: {e}')


def create_tfrecord_writer(file_counter, output_dir):
    tfrecord_path = os.path.join(output_dir, f'batch_{file_counter}.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecord_path)
    return writer


if __name__ == '__main__':
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn'
    file_dirs = [
        os.path.join(dbc_data_dir, 'label_train'), os.path.join(dbc_data_dir, 'label_val')]
    tfrecord_dirs = [
        os.path.join(dbc_data_dir, 'tfrecord/train'), os.path.join(dbc_data_dir, 'tfrecord/val')]
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
                writer = create_tfrecord_writer(file_counter, output_dir)
