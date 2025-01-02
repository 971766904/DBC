#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/14 19:22
# @Author  : zhongyu
# @Site    : 
# @File    : loop_change_dataset.py

'''
import pandas as pd
from jddb.file_repo import FileRepo
import numpy as np
import os
from jddb.processor import ShotSet, Shot
import h5py
import tensorflow as tf
from tfrecord_dataset import sample_generator, create_tfrecord_writer


def create_file_repo(round_num):
    # %%
    # load file_repo
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'

    training_shots = np.load(os.path.join(dbc_data_dir, 'info//dbc//dbc_training_shots_{}.npy'.format(round_num)))  # change file path
    valid_shots = np.load(os.path.join(dbc_data_dir, 'info//dbc//dbc_valid_shots_{}.npy'.format(round_num)))  # change file path

    # %%

    train_val_file_repo = FileRepo(os.path.join(dbc_data_dir, 'label_train+val//$shot_2$00//'))
    train_val_shot_list = train_val_file_repo.get_all_shots()
    train_val_shotset = ShotSet(train_val_file_repo, train_val_shot_list)

    # %%
    for shot in training_shots:
        shot_ob = Shot(shot, train_val_file_repo)
        shot_ob.save(FileRepo(os.path.join(dbc_data_dir, 'dbc//label_train_{}//$shot_2$00//'.format(round_num))),
                     data_type=float)  # change file path
    for shot in valid_shots:
        shot_ob = Shot(shot, train_val_file_repo)
        shot_ob.save(FileRepo(os.path.join(dbc_data_dir, 'dbc//label_val_{}//$shot_2$00//'.format(round_num))),
                     data_type=float)  # change file path

def create_tfrecord(round_num):
    dbc_data_dir = '/mypool/zy/dangerous/DBC/file_repo/data_file/processsed_data_cnn'
    file_dirs = [
        os.path.join(dbc_data_dir, 'dbc//label_train_{}'.format(round_num)),
        os.path.join(dbc_data_dir, 'dbc//label_val_{}'.format(round_num))]  # change file path
    tfrecord_dirs = [
        os.path.join(dbc_data_dir, 'dbc//tfrecord_{}/train'.format(round_num)),
        os.path.join(dbc_data_dir, 'dbc//tfrecord_{}/val'.format(round_num))]  # change file path
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

if __name__ == '__main__':
    for i in range(10):
        create_file_repo(i)
        create_tfrecord(i)
