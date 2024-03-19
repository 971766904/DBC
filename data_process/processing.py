#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/18 16:33
# @Author  : zhongyu
# @Site    : 
# @File    : processing.py

'''
#%%
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ResamplingProcessor, TrimProcessor
import pandas as pd

if __name__ == '__main__':
    #%%
    # load filerepo
    common_signal = []
    source_file_repo = FileRepo('H://rt//itu//FileRepo//processed_data_1k_5k_final//$shot_2$00//')
    train_file_repo = FileRepo('')

    #%%


