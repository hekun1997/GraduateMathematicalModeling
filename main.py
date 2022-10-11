import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.cluster import OPTICS
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score
from math import ceil

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

base_dir = r'D:\2021-2024\比赛活动\2022数学建模\2022题目\2022年中国研究生数学建模竞赛试题\2022年F题'
excel_1 = r'附件1：长春市COVID-19疫情期间病毒感染人数数据.xlsx'
excel_2 = r'附表2：长春市9个区隔离人口数量与生活物资投放点数量.xlsx'
excel_3 = r'附件3：长春市9个区交通网络数据和主要小区相关数据.xlsx'

areas = ['朝阳区', '南关区', '宽城区', '绿园区', '二道区', '长春新区(高新)', '经开区', '净月区', '汽开区']

excel_2_data = pd.read_excel(os.path.join(base_dir, excel_2))

split_numbers = [500, 1000, 1500, 2000, 5000, 10000, 20000, 50000, 100000]
cluster_number = [6, 5, 3, 4, 5, 4, 2, 3, 2]


def gmm_iter_n(area):
    area_index = excel_2_data[excel_2_data['区域名称'] == area].index.tolist()[0]
    area_data = excel_2_data.loc[area_index]
    area_people = float(area_data['隔离人口数（万人）']) * 10000

    df_3 = pd.read_excel(os.path.join(base_dir, excel_3), sheet_name='各区主要小区数据')

    area_index_df_3 = df_3[df_3['所属区域'] == area].index.tolist()
    start = min(area_index_df_3)
    end = max(area_index_df_3)
    coordinates = df_3[['小区横坐标', '小区纵坐标']].iloc[start: end].to_numpy()

    ss = []  # silhouette_score
    components = []

    best_labels = None
    best_ss = None
    for number in split_numbers:
        n_components = ceil(area_people / number)
        if n_components >= (end - start):
            n_components = (end - start)
        components.append(n_components)
        labels = GMM(n_components=n_components).fit_predict(coordinates)
        curr_ss = silhouette_score(coordinates, labels, metric='euclidean')
        ss.append(curr_ss)
        if curr_ss >= max(ss):
            best_labels = labels
            best_ss = curr_ss
    print('area: ' + area)
    print('cluster numbers: ' + str(len(set(best_labels))))
    print('best ss: ' + str(best_ss))
    print('silhouette_score: ' + str(ss))
    print('n_components: ' + str(components))
    # writer = pd.ExcelWriter(os.path.join(base_dir, '2\\' + area + '.xlsx'))
    #
    # df['label'] = best_labels.tolist()
    # df.to_excel(writer)
    # writer.save()


def gmm(area, n):
    df_3 = pd.read_excel(os.path.join(base_dir, excel_3), sheet_name='各区主要小区数据')

    area_index_df_3 = df_3[df_3['所属区域'] == area].index.tolist()
    start = min(area_index_df_3)
    end = max(area_index_df_3)

    coordinates = df_3[['小区横坐标', '小区纵坐标']].iloc[start: end].to_numpy()
    df = df_3.iloc[start: end]

    ss = []  # silhouette_score

    n_components = n
    labels = GMM(n_components=n_components).fit_predict(coordinates)
    curr_ss = silhouette_score(coordinates, labels, metric='euclidean')

    print('area: ' + area)
    print('cluster numbers: ' + str(len(set(labels))))
    print('curr ss: ' + str(curr_ss))
    print('silhouette_score: ' + str(ss))
    writer = pd.ExcelWriter(os.path.join(base_dir, '22\\' + area + '.xlsx'))

    df['label'] = labels.tolist()
    df.to_excel(writer)
    writer.save()


def mian1():
    for i, item in enumerate(areas):
        gmm(item, cluster_number[i])


def main2():
    for i, item in enumerate(areas):
        gmm_iter_n(item)


if __name__ == '__main__':
    # mian1()
    main2()
