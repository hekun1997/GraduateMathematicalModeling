import math
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REAL_SCALE = 332.664


def dis(WC, data):
    WCX = (np.array(data['x']) * WC).sum()
    WCY = (np.array(data['y']) * WC).sum()
    x0 = WCX / WC.sum()
    y0 = WCY / WC.sum()
    d_j = ((np.array(data['x']) - x0) ** 2 + (np.array(data['y']) - y0) ** 2) ** 0.5
    T = (WC * d_j).sum()
    print('重心法初始选点大致位置：（{}，{})'.format(x0, y0))
    print('总费用T0：{}'.format(T))

    # 迭代10次
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    for i in range(10):
        WC_j = WC / d_j
        WCX_j = ((np.array(data['x']) * WC) / d_j).sum()
        WCY_j = ((np.array(data['y']) * WC) / d_j).sum()
        x = WCX_j / WC_j.sum()
        y = WCY_j / WC_j.sum()
        d_j = ((np.array(data['x']) - x) ** 2 + (np.array(data['y']) - y) ** 2) ** 0.5
        T = (WC * d_j).sum()

    print('重心法最终选点大致位置：（{}，{})'.format(x, y))
    print('总费用T0：{}'.format(T))

    return [x0, y0], [x, y]


def with_label():
    base_dir = r'D:\2021-2024\比赛活动\2022数学建模\2022题目\2022年中国研究生数学建模竞赛试题\2022年F题\22'
    areas = ['朝阳区', '南关区', '宽城区', '绿园区', '二道区', '长春新区(高新)', '经开区', '净月区', '汽开区']

    xy = []
    center = []
    back_center = []
    f = open('f22.txt', 'w')
    for item in areas:
        print(item)
        file_path = os.path.join(base_dir, item + '.xlsx')
        df = pd.read_excel(file_path)
        walked_labels = []

        for i, row in df.iterrows():
            label = row['label']
            if label not in walked_labels:
                walked_labels.append(label)
                index = df[df['label'] == label].index.tolist()
                data = df.loc[index]
                people_count = data['小区人口数（人）'].sum()
                building_count = data['小区栋数'].sum()
                community_count = data['小区编号'].count()
                print('小区人口数: ' + str(people_count))
                print('小区栋数: ' + str(building_count))
                print('小区个数: ' + str(community_count))
                print('小区label: ' + str(label))
                input_data = data[['小区横坐标', '小区纵坐标']]
                input_data = input_data.rename(columns={'小区横坐标': 'x', '小区纵坐标': 'y'})
                w = np.array(data['小区人口数（人）'])
                init_point, final_point = dis(w, input_data)

                xy.append([np.array(input_data['x']), np.array(input_data['y'])])
                center.append(final_point)
                back_center.append(init_point)

                max_distance = 0
                for _i, _row in data.iterrows():
                    _x, _y = _row['小区横坐标'], _row['小区纵坐标']
                    _distance = math.sqrt(abs(_x - final_point[0]) ** 2 + abs(_y - final_point[1]) ** 2)
                    if _distance > max_distance:
                        max_distance = _distance
                R = max_distance * REAL_SCALE
                print('------管辖半径: {}, 小区label: {}-----'.format(R, label))
                # 编号&最优选址位置&备用选址位置&所属区域&选址半径&管辖人口数&管辖小区数
                content = '{} & ({:.5}, {:.5}) & ({:.5}, {:.5}) & {} & {:.5} & {} & {} \\\\'.format(label,
                                                                                                    final_point[0],
                                                                                                    final_point[1],
                                                                                                    init_point[0],
                                                                                                    init_point[1], item,
                                                                                                    R,
                                                                                                    people_count,
                                                                                                    community_count)
                print(content)
                f.write(content + '\n')
                print()

    f.close()
    colors = np.random.uniform(0, 1, (len(xy), 3))

    for i, v in enumerate(xy):
        plt.scatter(x=v[0], y=v[1], c=[colors[i]], alpha=0.5)
        plt.scatter(center[i][0], center[i][1], c=[colors[i]], marker='x')
        plt.scatter(back_center[i][0], back_center[i][1], c=[colors[i]], marker='*')

    plt.xlabel('小区横坐标')
    plt.ylabel('小区纵坐标')
    plt.savefig('大规模物资分拣场所选址.png', format='png', dpi=600)
    plt.show()


def no_label():
    base_dir = r'D:\2021-2024\比赛活动\2022数学建模\2022题目\2022年中国研究生数学建模竞赛试题\2022年F题\22'
    areas = ['朝阳区', '南关区', '宽城区', '绿园区', '二道区', '长春新区(高新)', '经开区', '净月区', '汽开区']

    xy = []
    center = []
    back_center = []
    f = open('f4.txt', 'w')
    for item in areas:
        print(item)
        file_path = os.path.join(base_dir, item + '.xlsx')
        df = pd.read_excel(file_path)

        people_count = df['小区人口数（人）'].sum()
        building_count = df['小区栋数'].sum()
        community_count = df['小区编号'].count()
        print('小区人口数: ' + str(people_count))
        print('小区栋数: ' + str(building_count))
        print('小区个数: ' + str(community_count))

        input_data = df[['小区横坐标', '小区纵坐标']]
        input_data = input_data.rename(columns={'小区横坐标': 'x', '小区纵坐标': 'y'})
        w = np.array(df['小区人口数（人）'])
        init_point, final_point = dis(w, input_data)

        xy.append([np.array(input_data['x']), np.array(input_data['y'])])
        center.append(final_point)
        back_center.append(init_point)

        # 最优选址位置&所属区域&选址半径&管辖人口数&管辖小区数
        content = '{} & ({:.5}, {:.5}) & {} & {} \\\\'.format(item, final_point[0], final_point[1], people_count, community_count)
        print(content)
        f.write(content + '\n')
        print()

    f.close()
    colors = np.random.uniform(0, 1, (len(xy), 3))

    for i, v in enumerate(xy):
        plt.scatter(x=v[0], y=v[1], c=[colors[i]], alpha=0.5)
        plt.scatter(center[i][0], center[i][1], c='r', marker='x')
        plt.scatter(back_center[i][0], back_center[i][1], c=[colors[i]], marker='*')

    plt.xlabel('小区横坐标')
    plt.ylabel('小区纵坐标')
    plt.savefig('网络上游.png', format='png', dpi=600)
    plt.show()


if __name__ == '__main__':
    no_label()
