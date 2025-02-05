import pandas as pd
import numpy as np
from collections import Counter

'''
    1.	从 CSV 文件中读取日志数据，并提取两部分信息：
	•	日志的“事件编号”（存放在 CSV 的第三列，索引为 2）
	•	日志的标签（存放在 CSV 的第一列，索引为 0），其中原始标签用 ‘-’ 表示正常，其它值表示异常
	2.	对日志数据进行预处理：
	•	对日志编号进行转换（去掉编号前的一个字符，并转成整数）
	•	将标签转换为 0 和 1，其中 0 表示正常，1 表示异常
	3.	利用滑动窗口（窗口长度为 300）对日志和标签数据进行切分：
	•	得到一系列长度为 300 的日志片段（每个片段为一个序列）
	•	同时，对应每个日志片段得到一个标签序列
	4.	根据每个窗口内是否包含异常（即是否存在标签 1），对整个窗口进行归类：
	•	如果窗口内任一位置为异常（1），则该窗口的最终标签记为 1；否则为 0
	5.	将处理后的日志序列和对应窗口标签分别保存为 CSV 文件，以便后续使用
	总结
	•	数据预处理：将日志事件从字符串转换为整数，并将原始标签由字符转换为 0（正常）或 1（异常）。
	•	滑动窗口构造：以固定长度 300 构造连续窗口，对日志序列和标签序列分别进行切片。
	•	窗口标签生成：如果一个窗口内至少存在一条异常日志，则该窗口整体标记为异常（1）；否则为正常（0）。
	•	保存结果：最终将窗口数据和对应标签分别保存为 CSV 文件，以便后续机器学习模型（如异常检测模型）的训练和评估。
    这段代码适用于处理大规模日志数据，通过滑动窗口技术构造时间序列数据，并将序列的异常情况归纳为一个整体标签，为后续基于序列的异常检测任务提供输入数据。
'''

sequence_length = 300
data = pd.read_csv('./bgl/BGL_100k_structured.csv')
data = data.values
pre_label = data[:,0]
logs = data[:,2]
# 处理datas
for i in range(0,len(logs)):
    logs[i] = int(logs[i][1:])
#处理label
label = []
for l in range(0,len(pre_label)):
    if pre_label[l]=='-':
        label.append(0)
    else:
        label.append(1)

logs_data = []
for j in range(len(logs) - sequence_length):
    logs_data.append(logs[j: j + sequence_length])
reshaped_logs = np.array(logs_data).astype('float64')

logs_label = []
for k in range(len(label) - sequence_length):
    logs_label.append(label[k: k + sequence_length])
reshaped_label = np.array(logs_label).astype('float64')
# reshaped_label = logs_label

result_label = []
for m in range(0,len(reshaped_label)):
    if 1 in reshaped_label[m]:
        result_label.append(1)
    else:
        result_label.append(0)
end_logs = []
end_label = []
for n in range(0,len(result_label)):
    # if n%10 == 0:
    end_logs.append(reshaped_logs[n])
    end_label.append(result_label[n])
print(Counter(end_label))

pd.DataFrame(data=end_logs).to_csv('../data/bgl_data.csv', index=False, header=False)
pd.DataFrame(data=end_label).to_csv('../data/bgl_label.csv', index=False, header=False)