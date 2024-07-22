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


# 城市坐标
# city_pos_list = np.random.rand(config.city_num, config.pos_dimension)

# 替换 city_pos_list 为固定的坐标数组
city_pos_list = np.array([
    [67.50, 10.11, 29.33],
    [77.26, 69.57, 74.74],
    [83.38, 22.45, 79.39],
    [76.35, 50.75, 54.87],
    [40.42, 18.64, 73.29],
    [73.76, 26.53, 43.98],
    [4.92, 16.17, 93.49],
    [95.77, 86.62, 14.55],
    [55.32, 11.48, 27.65],
    [89.38, 27.58, 77.86]
])

# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)
# 确认数组的实际形状和大小
print("city_pos_list shape:", city_pos_list.shape)  # 应为 (10, 3)
print("Length of city_pos_list:", len(city_pos_list))  # 应为 10
print("city_dist_mat shape:", city_dist_mat.shape)

print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
result_pos_list = city_pos_list[result, :]

# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
print(matplotlib.matplotlib_fname())

if len(result_pos_list[0]) == 3:
    # 原始图片
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维散点图
    ax.plot(result_pos_list[:, 0], result_pos_list[:, 1], result_pos_list[:, 2], 'o-r')
    # 添加标题
    ax.set_title(u"三维路线")
    # 添加标签
    ax.set_xlabel('X 轴')
    ax.set_ylabel('Y 轴')
    ax.set_zlabel('Z 轴')
    plt.show()
    # 原始图片

    fig = go.Figure()
    # 添加散点图
    fig.add_trace(go.Scatter3d(
        x=result_pos_list[:, 0],
        y=result_pos_list[:, 1],
        z=result_pos_list[:, 2],
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=5, color='blue')
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()
else:
    fig = plt.figure()
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
    plt.title(u"路线")
    plt.legend()
    fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"适应度曲线")
plt.legend()
fig.show()
