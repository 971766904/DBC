#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/16 22:19
# @Author  : zhongyu
# @Site    : 
# @File    : read_data.py

'''
from MDSplus import connection
import numpy as np
import warnings

c = connection.Connection('222.20.94.136')


def read_data_from_tree(shot, tag):
    c.openTree('jtext', shot=shot)
    try:
        tag_data = np.array(c.get(tag))  # diagnostic data
        tag_time = np.array(c.get(r'DIM_OF(BUILD_PATH({}))'.format(tag)))  # DIM_OF(tag), time axis
    except Exception as e:
        warnings.warn("Could not read data from {}".format(shot), category=UserWarning)
    c.closeAllTrees()
    return tag_data, tag_time
