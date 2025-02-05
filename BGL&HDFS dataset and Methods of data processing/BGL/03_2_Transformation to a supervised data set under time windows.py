import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import csv
import copy
from collections import Counter


'''
    这段代码的主要作用和流程为：
	1.	读取数据：从 BGL_sequence.csv 中读取包含事件序列和标签的原始数据。
	2.	序列清洗：
	•	对于每条序列，先以逗号分隔成多个子串；
	•	判断序列是否为空（用 '[]' 判断）；
	•	对每个子串，根据是否包含 [ 或 ] 进行字符串切片，提取出中间的数字部分，并转换为整数；
	•	将得到的数字序列转换为字符串（去除最外层的中括号）后存入列表。
	3.	标签处理：将原始标签转换为整数，并保存到另一个列表。
	4.	统计与保存：
	•	输出成功解析和空序列的数量；
	•	将清洗后的序列和对应标签写入新的 CSV 文件，以供后续使用（例如作为模型输入或进一步的数据分析）。

    通过这种处理方式，可以将原始带有多余符号的事件序列转变为标准化的数值序列，同时保留每个序列的异常或正常标记，为后续的日志分析或异常检测任务提供干净的数据。
'''

pre_data = pd.read_csv('./bgl/BGL_sequence.csv')
pre_data = pre_data.values

data = []
label = []
count = 0
co = 0
for i in range(0,len(pre_data)):
    value = []
    division = pre_data[i][0].split(",")
    if division[0] != '[]':
        for j in range(0,len(division)):
            if '[' in division[j] and ']' not in division[j]:
                value.append(int(division[j][3:-1]))
            elif '[' in division[j] and ']' in division[j]:
                value.append(int(division[j][3:-2]))
            elif '[' not in division[j] and ']' in division[j]:
                value.append(int(division[j][3:-2]))
            else:
                value.append(int(division[j][3:-1]))
        line = str(np.array(value))[2:-1].split(' ')
        data.append(str(np.array(value))[1:-1])
        label.append(int(pre_data[i][1]))
        co = co+1
    else:
        count = count+1
print(co)
print(count)

pd.DataFrame({'Sequence':data,'label':label}).to_csv('./data/bgl_time_data.csv',index=False, header=False)
