import matplotlib.pyplot as plt
import numpy as np

# 数据
categories = ['Precision', 'Recall', 'F-measure']
# deplog = [92.87, 95.49, 94.16]
# loganomaly = [96.15, 96.28, 96.21]
# robustlog = [96.44, 91.61, 93.96]
# lightlog = [94.37, 99.78, 97.0]


deplog = [0.92, 0.95, 0.93]
loganomaly = [0.96, 0.96, 0.96]
robustlog = [0.96, 0.91, 0.93]
ourMethod = [0.94, 0.99, 0.97]

# 位置参数
x = np.arange(len(categories))
width = 0.2

# 选择更浅的颜色
colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']  # 浅蓝色, 浅橙色, 浅绿色, 浅红色

# 画图
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - 1.5 * width, deplog, width, label='DeepLog', color=colors[0])
rects2 = ax.bar(x - 0.5 * width, loganomaly, width, label='LogAnomaly', color=colors[1])
rects3 = ax.bar(x + 0.5 * width, robustlog, width, label='RobustLog', color=colors[2])
rects4 = ax.bar(x + 1.5 * width, ourMethod, width, label='OurMethod', color=colors[3])

# 设定标签
ax.set_xlabel('HDFS Dataset')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 显示数值
for rects in [rects1, rects2, rects3, rects4]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

plt.show()