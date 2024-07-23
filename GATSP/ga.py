import config as conf
import random

city_dist_mat = None
config = conf.get_config()
# 各项参数
gene_len = config.city_num
individual_num = config.individual_num
gen_num = config.gen_num
mutate_prob = config.mutate_prob


def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类
class Individual:
    def __init__(self, genes=None):
        # 随机生成序列
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0.0
        for i in range(gene_len - 1): #遍历所有城市的序号，gene-len就是城市数量
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += city_dist_mat[from_idx, to_idx] #适应度就是距离
        # 连接首尾
        fitness += city_dist_mat[self.genes[-1], self.genes[0]]
        #负数索引通常用来表示从列表末尾开始的元素。
        return fitness


class Ga:
    def __init__(self, input_):
        global city_dist_mat
        city_dist_mat = input_
        self.best = None  # 每一代的最佳个体
        self.individual_list = []  # 每一代的个体列表 个体都具有适应度和基因gene
        self.result_list = []  # 每一代对应的解
        self.fitness_list = []  # 每一代对应的适应度

    def cross(self):
        new_gen = []
        random.shuffle(self.individual_list) # 对种群中个体的随机化排序。
        for i in range(0, individual_num - 1, 2): # 表示序列中每个元素之间的步长为2，i的取值会依次为0, 2, 4,
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, gene_len - 2)
            index2 = random.randint(index1, gene_len - 1) # 选择基因的一个区间，i1到i2；
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # pos1_recorder是一个字典，其中键是genes1列表中的元素，而值是该元素在列表中的索引位置。
            # 交叉
            for j in range(index1, index2):
                # 1  2 3 5  6
                # 3  1 2 6  5
                value1, value2 = genes1[j], genes2[j] # 2  1
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1] # j=2时，pos1为1，pos2 3
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j] # 快速交换变量值
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j] # 交换了genes1列表中索引为j和pos1位置的元素值。
                # 2  1 3 5  6
                # 3  2 1 6  5
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j #将 value1，2的索引位置更新为pos1 1，将value2 1的索引位置更新为j 2。
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(genes1))
            new_gen.append(Individual(genes2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < mutate_prob:
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, gene_len - 2)
                index2 = random.randint(index1, gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
                # 将经过翻转的基因片段genes_mutate插入回个体的基因序列中，从而生成新的个体。
        # 两代合并
        self.individual_list += new_gen

    def select(self):
        # 锦标赛
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = individual_num // group_num  # 每小组获胜人数 individual_num 这里设置的为60.每个小组获胜6个人。
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:group_winner] # 选出前几个适应度高的。适应度排序使用的升序排列。
        self.individual_list = winners

    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        # 产生下一代的方法
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # 获得这一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        # 初代种群
        self.individual_list = [Individual() for _ in range(individual_num)]
        # 一个包含individual_num个Individual对象的列表
        # 每个Individual对象都是通过不带参数的Individual构造函数创建的，因此会随机生成一个基因序列。
        self.best = self.individual_list[0] # 记录每一代中的最佳解
        # 迭代
        for i in range(gen_num):
            self.next_gen()
            # 连接首尾
            result = copy_list(self.best.genes)
            result.append(result[0]) #最后加上第一个个体
            self.result_list.append(result) #每一代的最优的路径的结果图
            self.fitness_list.append(self.best.fitness) #每一代的适应度
        return self.result_list, self.fitness_list
