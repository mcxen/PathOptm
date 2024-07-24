import numpy as np
import config as conf
from ga import Ga
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import matplotlib
config = conf.get_config()

# 计算距离矩阵
def build_dist_mat(input_list):
    n = len(input_list)  # 使用实际的长度而不是 config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            distance = np.linalg.norm(d) #使用欧几里得距离，既可以计算二维的距离又可以计算三维的距离
            dist_mat[i, j] = distance
            dist_mat[j, i] = distance
    return dist_mat


def plot_3d_path(result_pos_list, title='Path Optimization'):
    # 确保输入数据是numpy数组
    result_pos_list = np.array(result_pos_list)

    # 使用 Matplotlib 绘制三维路线
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


# 城市坐标
# city_pos_list = np.random.rand(config.city_num, config.pos_dimension)

# 替换 city_pos_list 为固定的坐标数组
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
# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)
# 确认数组的实际形状和大小
print("city_pos_list shape:", city_pos_list.shape)  # 应为 (10, 3)
print("Length of city_pos_list:", len(city_pos_list))  # 应为 10
print("city_dist_mat shape:", city_dist_mat.shape)

# print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
result_pos_list = city_pos_list[result, :]
# print(result_list)
# 绘图
# # 解决中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# print(matplotlib.matplotlib_fname())

if len(result_pos_list[0]) == 3:
    plot_3d_path(result_pos_list)
else:
    fig = plt.figure()
    print(result_pos_list)
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
    plt.title(u"Tsp Path")
    plt.legend()
    fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"Fitness Condition")
plt.legend()
fig.show()
