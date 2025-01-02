#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/16 12:38
# @Author  : zhongyu
# @Site    : 
# @File    : name_shot_tfrecord.py

'''
import os
import h5py
import tensorflow as tf
from jddb.file_repo import FileRepo
from jddb.processor import Shot


def sample_generator(file_paths):
    try:
        with h5py.File(file_paths, 'r') as h5_file:
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
        print(f'current file: {file_paths}\nException: {e}')


def create_tfrecord_writer(output_dir, shot):
    tfrecord_path = os.path.join(output_dir, f'{shot}.tfrecord')
    writer = tf.io.TFRecordWriter(tfrecord_path)
    return writer


def write_tfrecord(input_dir, output_dir, shot_list):
    os.makedirs(output_dir, exist_ok=True)
    for each_shot in shot_list:
        file_path = os.path.join(input_dir, f'{each_shot}.hdf5')
        sample_gen = sample_generator(file_path)
        writer = create_tfrecord_writer(output_dir, each_shot)

        for sample in sample_gen:
            feature_dict = dict()

            for key, value in sample.items():
                feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))
                feature_dict[key] = feature

            features = tf.train.Features(feature=feature_dict)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
        writer.close()


if __name__ == '__main__':
    print('tfrecord')
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    train_val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//'))
    shot_list = train_val_file_repo.get_all_shots()
    for shot in shot_list:
        shot_ob = Shot(shot, train_val_file_repo)
        shot_ob.save(FileRepo(os.path.join(dbc_data_dir, 'label_train_val_no_sub')),
                     data_type=float)
    write_tfrecord(input_dir=os.path.join(dbc_data_dir, 'label_train_val_no_sub'),
                   output_dir=os.path.join(dbc_data_dir, 'tfrecord_train+val'), shot_list=shot_list)
