#!/usr/bin/env python
# encoding: utf-8
'''
# @Time    : 2023/9/30 10:00
# @Author  : zhongyu
# @Site    : 
# @File    : fig_plot.py

'''
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


def get_data_time(file_repo, shot, tags):
    """
    get data and time from file_repo with tags, return a dict contain all tags' data and time.
    @param file_repo:
    @param shot:
    @param tags:
    @return: a dict whose key is tag of tags and value is a 2d array, the first row is data and second is time.
    """
    data_dict = file_repo.read_data(shot, tags)
    for tag in tags:
        attribute_tag = file_repo.read_attributes(shot, tag, ['SampleRate', 'StartTime'])
        time = np.arange(len(data_dict[tag])) / attribute_tag['SampleRate'] + attribute_tag['StartTime']
        data = data_dict[tag].reshape(1, -1)
        time = time.reshape(1, -1)
        data_dict[tag] = np.append(data, time, axis=0)
    return data_dict


class Plot2d:
    """
    plot 2d array diagnose fig with file_repo, array tag, shot num.
    """
    def __init__(self, array_t, shot, file_repo):
        """

        @param array_t:
        @param shot:
        @param file_repo:
        """
        self.array_t = array_t
        self.shot = shot
        self.file_repo = file_repo

    def array_plot(self):
        """
        plot 2d array diagnose fig
        """
        disrup_label = self.file_repo.read_labels(self.shot, ['IsDisrupt'])['IsDisrupt']
        data_dict = get_data_time(self.file_repo, self.shot, self.array_t)
        local = np.arange(len(self.array_t)) + 1
        array_1 = np.empty([0, data_dict[self.array_t[0]].shape[1]])
        time = data_dict[self.array_t[0]][1]
        for tag in self.array_t:
            data = data_dict[tag][0].reshape(1, -1)
            array_1 = np.concatenate((array_1, data))
        plt.figure()
        plt.contourf(time, local, array_1)
        plt.colorbar()
        plt.xlabel('time(s)')
        plt.title('#{0} ({1})'.format(self.shot, disrup_label))
        plt.show()
        plt.savefig('_temp_view_fig/{}_pxuv.png'.format(self.shot), bbox_inches='tight')
        plt.close()


class Plotsignal:
    def __init__(self, file_repo, tag_list, shot):
        self.file_repo = file_repo
        self.tag_list = tag_list
        self.shot = shot

    def signal_plot(self):
        disrup_label = self.file_repo.read_labels(self.shot, ['IsDisrupt'])['IsDisrupt']
        data_dict = get_data_time(self.file_repo, self.shot, self.tag_list)
        fig, axes = plt.subplots(nrows=int(len(self.tag_list) / 2), ncols=1, sharex=True)
        fig.suptitle('#{} ({})'.format(self.shot,disrup_label))
        # 微软雅黑,如果需要宋体,可以用simsun.ttc
        myfont = fm.FontProperties(fname='C:/Windows/Fonts/simsun.ttc', size=12)
        fronts = {'family': 'Times New Roman', 'size': 12}
        for i in range(0, len(self.tag_list), 2):
            data = data_dict[self.tag_list[i]][0]
            time = data_dict[self.tag_list[i]][1]
            data_t = data_dict[self.tag_list[i + 1]][0]
            time_t = data_dict[self.tag_list[i + 1]][1]

            axes1 = axes[int(i / 2)]
            lns1 = axes1.plot(time, data, 'r', label=self.tag_list[i])
            # plt.axvline(x=2,c = 'yellow')
            axes1.set_ylabel(self.tag_list[i], fontproperties=myfont)
            # axes1.set_yticks(fontproperties='Times New Roman', fontsize=12)
            axes2 = axes1.twinx()
            lns2 = axes2.plot(time_t, data_t, 'b', label=self.tag_list[i + 1])
            axes2.set_ylabel(self.tag_list[i + 1], fontproperties=myfont)
            # 合并图例
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            axes2.legend(lns, labs)
        axes1.set_xlabel('time(s)', fontproperties=myfont)

        plt.show()
        plt.savefig('_temp_view_fig/{}_basic.png'.format(self.shot), bbox_inches='tight')
        plt.close()
