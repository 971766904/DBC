#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2024/3/7 16:40
# @Author  : zhongyu
# @Site    : 
# @File    : consecutive_shot.py

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jddb.file_repo import FileRepo

if __name__ == '__main__':
    # sort shots in consecutive way
    # read shot list csv file
    source_file_repo = FileRepo('..//file_repo//$shot_2$00//')
    shot_list = source_file_repo.get_all_shots()
    shots_infor = pd.read_csv('..//file_repo//info//info.csv')
    shots_infor = shots_infor.sort_values(by='shot')