import matplotlib.pyplot as plt
import numpy as np

# 数据，按照每个方法的Precision, Recall, F1来组织
methods = ['DeepLog', 'LogAnomaly', 'RobustLog', 'OurMethod']
precision = [0.92, 0.96, 0.96, 0.94]
recall = [0.95, 0.96, 0.91, 0.99]
f1 = [0.93, 0.96, 0.93, 0.97]

# 位置参数
x = np.arange(len(methods))
width = 0.2

# 选择颜色
colors = ['#aec7e8', '#ffbb78', '#98df8a']  # 浅蓝色, 浅橙色, 浅绿色

# 创建画图
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制每个方法的Precision, Recall, F1
rects1 = ax.bar(x - width, precision, width, label='Precision', color=colors[0])
rects2 = ax.bar(x, recall, width, label='Recall', color=colors[1])
rects3 = ax.bar(x + width, f1, width, label='F1', color=colors[2])

# 设置标签
ax.set_xlabel('Methods')
ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

# 显示数值
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

plt.show()