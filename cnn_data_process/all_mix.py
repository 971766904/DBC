#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/4/10 20:36
# @Author  : zhongyu
# @Site    : 
# @File    : all_mix.py

'''
from jddb.file_repo import FileRepo
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from jddb.processor import ShotSet
from util.basic_processor import CutProcessor, BinaryLabelProcessor
import os
import h5py
import tensorflow as tf
from tfrecord_dataset import sample_generator, create_tfrecord_writer


def spare_dataset_and_record_dbc(source_file_repo, dbc_data_dir, training_shots, valid_shots, test_shots):
    shots_list = source_file_repo.get_all_shots()
    processed_shotset = ShotSet(source_file_repo, shots_list)

    # spare dataset and record dbc
    train_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=training_shots,
                                                    save_repo=FileRepo(
                                                        os.path.join(dbc_data_dir, 'remove_train//$shot_2$00//')))
    val_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=valid_shots,
                                                  save_repo=FileRepo(
                                                      os.path.join(dbc_data_dir, 'remove_val//$shot_2$00//')))
    test_shotset = processed_shotset.remove_signal(tags=['stacked_data'], keep=True, shot_filter=test_shots,
                                                   save_repo=FileRepo(
                                                       os.path.join(dbc_data_dir, 'remove_test//$shot_2$00//')))

    # cut dataset
    train_shotset = train_shotset.process(processor=CutProcessor(pre_time=20, is_test=False),
                                          input_tags=['stacked_data'],
                                          output_tags=['stacked_data'],
                                          shot_filter=training_shots,
                                          save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_train//$shot_2$00//')))
    val_shotset = val_shotset.process(processor=CutProcessor(pre_time=20, is_test=False),
                                      input_tags=['stacked_data'],
                                      output_tags=['stacked_data'],
                                      shot_filter=valid_shots,
                                      save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_val//$shot_2$00//')))
    test_shotset = test_shotset.process(processor=CutProcessor(pre_time=20, is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['stacked_data'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'cut_test//$shot_2$00//')))

    # label
    train_shotset = train_shotset.process(processor=BinaryLabelProcessor(is_test=False),
                                          input_tags=['stacked_data'],
                                          output_tags=['label'],
                                          shot_filter=training_shots,
                                          save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_train//$shot_2$00//')))
    val_shotset = val_shotset.process(processor=BinaryLabelProcessor(is_test=False),
                                      input_tags=['stacked_data'],
                                      output_tags=['label'],
                                      shot_filter=valid_shots,
                                      save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_val//$shot_2$00//')))
    test_shotset = test_shotset.process(processor=BinaryLabelProcessor(is_test=True),
                                        input_tags=['stacked_data'],
                                        output_tags=['label'],
                                        shot_filter=test_shots,
                                        save_repo=FileRepo(os.path.join(dbc_data_dir, 'label_test//$shot_2$00//')))


def make_tfrecord_dataset(dbc_data_dir):
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


if __name__ == '__main__':
    # %%
    # load file repo
    dbc_data_dir = '..//..//file_repo//data_file//processed_data_cnn//all_mix'
    source_file_repo = FileRepo('..//..//file_repo//data_file//slice//$shot_2$00//')

    # %%
    # spare dataset and record dbc
    shot_list = source_file_repo.get_all_shots()
    # disruption tag for dataset split
    is_disrupt = []
    for shot in shot_list:
        dis_label = source_file_repo.read_labels(shot, ['IsDisrupt'])
        is_disrupt.append(dis_label['IsDisrupt'])
    print('all shots:{}'.format(len(shot_list)))
    print('disruption shots:{}'.format(sum(is_disrupt)))
    # train test split on shot not sample according to whether shots are disruption
    # set test_size=0.5 to get 50% shots as test set
    train_shots, test_shots, train_label, _ = \
        train_test_split(shot_list, is_disrupt, test_size=0.2,
                         random_state=1, shuffle=True, stratify=is_disrupt)
    train_shots, val_shots, _, _ = \
        train_test_split(train_shots, train_label, test_size=0.2,
                         random_state=1, shuffle=True, stratify=train_label)
    # save the split result in npy file
    np.save(os.path.join(dbc_data_dir, 'train_shots.npy'), train_shots)
    np.save(os.path.join(dbc_data_dir, 'val_shots.npy'), val_shots)
    np.save(os.path.join(dbc_data_dir, 'test_shots.npy'), test_shots)

    # %%
    # processing
    spare_dataset_and_record_dbc(source_file_repo, dbc_data_dir, train_shots, val_shots, test_shots)

    # %%
    # make tfrecord dataset
    make_tfrecord_dataset(dbc_data_dir)
