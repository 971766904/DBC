#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/12/26 21:25
# @Author  : zhongyu
# @Site    : 
# @File    : check_range.py

'''
from jddb.file_repo import FileRepo
from jddb.processor import ShotSet
from jddb.processor.basic_processors import ResamplingProcessor, NormalizationProcessor, TrimProcessor
from util.basic_processor import find_tags, read_config, StackProcessor, CutProcessor, BinaryLabelProcessor, \
    SliceProcessor, WindStackProcessor, OldSliceProcessor
from sklearn.model_selection import train_test_split
import json

if __name__ == '__main__':
    pass