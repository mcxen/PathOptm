# 遗传算法求解TSP问题

交叉算法：

```txt
# 1  2 3 5  6
# 3  1 2 6  5

基因片段： 2 3 5，1 2 6

首先交换里面的，2 1
2 1 3 5 6
3 2 1 6 5

把下面的1 和3交换位置，直接就是通过交换数组的1 3 位置实现的
2 3 1 5 6
1 2 3 6 5

接下来交换5 6
2 3 1 6 5
1 2 3 5 6
结果如上，避免了交换了重复的基因片段
# 1  2 3 5  6
# 3  1 2 6  5
=》
# 1  1 2 6  6
# 3  2 3 5  5
这样的情况
```



```python
def cross(self):
    new_gen = []
    random.shuffle(self.individual_list) # 对种群中个体的随机化排序。
    for i in range(0, individual_num - 1, 2): 
      # 表示序列中每个元素之间的步长为2，i的取值会依次为0, 2, 4,
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
            genes2[j], genes2[pos2] = genes2[pos2], genes2[j] 
            # 交换了genes1列表中索引为j和pos1位置的元素值。
            # 2  1 3 5  6
            # 3  2 1 6  5
            pos1_recorder[value1], pos1_recorder[value2] = pos1, j 
            #将 value1，2的索引位置更新为pos1 1，将value2 1的索引位置更新为j 2。
            pos2_recorder[value1], pos2_recorder[value2] = j, pos2
        new_gen.append(Individual(genes1))
        new_gen.append(Individual(genes2))
    return new_gen
```





### 三维求解结果示意图

<img src="./assets/README/CleanShot 2024-07-23 at 08.54.15@2x.png" alt="CleanShot 2024-07-23 at 08.54.15@2x" style="zoom:50%;" />