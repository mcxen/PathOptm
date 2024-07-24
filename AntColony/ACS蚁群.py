# -*- coding: utf-8 -*-
import random
import copy
import sys
import math
import tkinter #//GUI模块
import threading#//多线程编程
import time
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
(ALPHA, BETA, RHO, Q) = (1.5,2.0,0.9,100.0)

#----------- 蚂蚁 -----------
class Ant(object):
    # 初始化
    def __init__(self,ID):
        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点
    # 初始数据
    def __clean_data(self):
        self.path = []               # 当前蚂蚁的路径
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)] # 探索城市的状态
        
        city_index = random.randint(0,city_num-1) # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index) #路径加上当前的idx
        self.open_table_city[city_index] = False
        self.move_count = 1
    
    # 选择下一个城市
    def __choice_next_city(self):
        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  #存储去下个城市的概率
        total_prob = 0.0
        # 对于每一个城市（city_num 表示城市的数量）
        for i in range(city_num):
            # 如果城市 i 还未被访问（open_table_city 是一个标记已访问城市的数组）
            if self.open_table_city[i]:
                try:
                    # 计算从当前城市到城市 i 的选择概率
                    # select_citys_prob 是一个存储选择概率的数组
                    # pheromone_graph 是信息素浓度矩阵
                    # distance_graph 是距离矩阵
                    # ALPHA 和 BETA 是控制信息素浓度和距离影响力的参数
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        (1.0 / distance_graph[self.current_city][i]), BETA)
                    # 累加总概率（total_prob 用于后续的归一化）
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    # 捕获除零错误，如果发生这种错误，打印蚂蚁的ID、当前城市和目标城市的信息
                    # 然后退出程序（sys.exit(1)）
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID=self.ID,
                                                                                                current=self.current_city,
                                                                                                target=i))
                    sys.exit(1)
        # 轮盘选择城市
        # 如果总概率大于0
        if total_prob > 0.0:
            # 生成一个在0.0到total_prob之间的随机概率
            temp_prob = random.uniform(0.0, total_prob)
            # 遍历所有城市
            for i in range(city_num):
                # 如果城市i还未被访问
                if self.open_table_city[i]:
                    # 从临时概率中减去城市i的选择概率
                    temp_prob -= select_citys_prob[i]
                    # 如果临时概率小于0
                    if temp_prob < 0.0:
                        # 当随机数小于0时，表明当前城市的累积概率区间包含该随机数，**因此选择当前城市是合理的。**
                        next_city = i
                        # 结束循环
                        break
        # 如果轮盘赌选择后 next_city 仍然是 -1
        if (next_city == -1):
            # 随机选择一个城市
            next_city = random.randint(0, city_num - 1)
            # 如果选择的城市已经被访问过
            while ((self.open_table_city[next_city]) == False):
                # 再随机选择一个城市
                next_city = random.randint(0, city_num - 1)
        # 返回下一个城市序号
        return next_city
    
    # 计算路径总距离
    def __cal_total_distance(self):
        
        temp_distance = 0.0
 
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += distance_graph[start][end]
 
        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance
        
    
    # 移动操作
    def __move(self, next_city):
        
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1
        
    # 搜索路径
    def search_path(self):
 
        # 初始化数据
        self.__clean_data()
 
        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city =  self.__choice_next_city()
            self.__move(next_city)
 
        # 计算路径总长度
        self.__cal_total_distance()
        # print(self.total_distance)
 
#----------- TSP问题 -----------
class TSPPLT(object):
    def __init__(self, n_ants, n_iterations):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.ants = [Ant(ID=i) for i in range(n_ants)]
        self.best_ant = copy.deepcopy(self.ants[0])
        self.iter = 0
        self.__lock = threading.Lock()
        self.__running = False
        self.fitness_list = []  # 每一代对应的适应度

    def search_path(self, evt=None):
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running and self.iter < self.n_iterations:
            for ant in self.ants:
                ant.search_path()
                # print(self.best_ant.total_distance)
                if self.best_ant.total_distance ==0:
                    self.best_ant.total_distance =1<<31
                if ant.total_distance < self.best_ant.total_distance:
                    self.best_ant = copy.deepcopy(ant)
            self.__update_pheromone_graph()
            print(f"迭代次数：{self.iter} 最佳路径总距离：{int(self.best_ant.total_distance)}")
            self.fitness_list.append(self.best_ant.total_distance)
            # self.plot_path(self.best_ant.path)
            self.iter += 1
        return self.fitness_list
    def __update_pheromone_graph(self):
        temp_pheromone = np.zeros((city_num, city_num))
        for ant in self.ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                temp_pheromone[start][end] += Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]

        global pheromone_graph
        # 更新所有城市的信息素，旧的信息素衰减加上新的信息素。
        pheromone_graph = pheromone_graph * RHO + temp_pheromone

    def plot_path(self, path):
        path_coords = [city_pos_list[city] for city in path]
        path_coords.append(path_coords[0])  # 回到起点
        path_coords = np.array(path_coords)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(city_pos_list[:, 0], city_pos_list[:, 1], city_pos_list[:, 2], c='red', label='Cities')
        ax.plot(path_coords[:, 0], path_coords[:, 1], path_coords[:, 2], c='blue', label='Path')

        for i, pos in enumerate(city_pos_list):
            ax.text(pos[0], pos[1], pos[2], f'{i}', fontsize=12, ha='right')

        ax.set_title(f'TSP Path Opt, Total Distance: {self.best_ant.total_distance:.2f}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.legend()
        plt.show()

def plot_3d_path(result_pos_list, title='Path Optm'):
    # 使用 Matplotlib 绘制三维路线
    # 确保输入数据是numpy数组
    result_pos_list = np.array(result_pos_list)
    def plot_with_matplotlib(result_pos_list):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(result_pos_list[:, 0], result_pos_list[:, 1], result_pos_list[:, 2], 'o-r')
        ax.set_title(title)
        ax.set_xlabel('X 轴')
        ax.set_ylabel('Y 轴')
        ax.set_zlabel('Z 轴')
        plt.show()

    # 使用 Plotly 绘制三维路线
    def plot_with_plotly(result_pos_list):
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=result_pos_list[:, 0],
            y=result_pos_list[:, 1],
            z=result_pos_list[:, 2],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=5, color='blue')
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        fig.show()

    # 调用 Matplotlib 绘制
    plot_with_matplotlib(result_pos_list)

    # 调用 Plotly 绘制
    plot_with_plotly(result_pos_list)
#----------- 程序的入口处 -----------
                
if __name__ == '__main__':
 
    print (u""" 
--------------------------------------------------------
    程序：蚁群算法解决TPS问题程序 
-------------------------------------------------------- 
    """)

    # 城市坐标列表
    # city_pos_list = np.array([
    #     [67.50, 10.11, 29.33],
    #     [77.26, 69.57, 74.74],
    #     [83.38, 22.45, 79.39],
    #     [76.35, 50.75, 54.87],
    #     [40.42, 18.64, 73.29],
    #     [73.76, 26.53, 43.98],
    #     [4.92, 16.17, 93.49],
    #     [95.77, 86.62, 14.55],
    #     [55.32, 11.48, 27.65],
    #     [89.38, 27.58, 77.86]
    # ])
    city_pos_list = np.array([
        [50.5, 109.3], [55.7, 100.7], [68.7, 101.9], [73.8, 100.7], [79.7, 109.8],
        [93.2, 101.6], [112.2, 103.1], [108.1, 111.7], [40.4, 132], [67.1, 133.2],
        [67.6, 148.1], [60.3, 160.4], [49.6, 161.6], [42.5, 147.4], [33.3, 147.2],
        [31.8, 158.5], [78.9, 165], [91.3, 164.8], [99.3, 149.8], [107.6, 163.3],
        [121.1, 168.1], [131.8, 150.3], [135.5, 169.3], [150.3, 171.5], [162.4, 160.9],
        [162.9, 152.5], [123.3, 134.6], [94.3, 133.4], [147.1, 136.3], [174, 135.8],
        [172.8, 110.8], [200.1, 135.8], [198.6, 123.3], [230.9, 133.9], [224.3, 136.6],
        [222.3, 153.2], [163.9, 170.5], [180.4, 175.1], [190.1, 176.1], [210.2, 178],
        [220.7, 179.2], [240.8, 181.9], [250.8, 185], [247.4, 155.8], [272.4, 156.8],
        [270.4, 176.6], [271.6, 183.8], [296.9, 176.6], [301.2, 187.6], [300.7, 156.6],
        [239.9, 88.4], [218, 84.5], [197.4, 83.8], [174.8, 81.1], [153.5, 79.5],
        [122.1, 76.6], [225.1, 57.8], [236, 74.6], [121.2, 72.2], [98.1, 75.1]
    ])

    # 城市数量
    city_num = city_pos_list.shape[0]

    # 定义信息素矩阵和距离矩阵
    pheromone_graph = np.random.rand(city_num, city_num)
    distance_graph = np.zeros((city_num, city_num))

    # 计算城市之间的欧几里得距离
    for i in range(city_num):
        for j in range(city_num):
            if i != j:
                distance_graph[i][j] = np.linalg.norm(city_pos_list[i] - city_pos_list[j])

    # 定义常量
    ALPHA = 1.0
    BETA = 5.0
    RHO = 0.5  # 信息素挥发因子
    Q = 100.0  # 信息素常量

    aco = TSPPLT(n_ants=60, n_iterations=100)
    fitness_list = aco.search_path()
    # aco.plot_path(aco.best_ant.path)
    path_coords = [city_pos_list[city] for city in aco.best_ant.path]
    path_coords.append(path_coords[0])  # 回到起点
    path_coords = np.array(path_coords)
    if len(path_coords[0])==3:
        plot_3d_path(path_coords)
    else:
        fig = plt.figure()
        # print(path_coords)
        plt.plot(path_coords[:, 0], path_coords[:, 1], 'o-r')
        plt.title(u"Tsp Path")
        plt.legend()
        fig.show()
    fig = plt.figure()
    plt.plot(fitness_list)
    plt.title(u"Fitness Condition")
    plt.legend()
    fig.show()
