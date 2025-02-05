import pandas as pd
import numpy as np
from collections import Counter

'''
    总结
	1.	读取和转换数据：从 CSV 文件读取数据，并将其转换为 NumPy 数组进行后续操作。
	2.	打乱数据顺序：通过多次调用 np.random.shuffle 打乱数据顺序，增加数据的随机性。
	3.	数据筛选：从打乱后的数据中提取前 50000 条记录，用于后续分析或模型训练。
	4.	标签分布统计：使用 Counter 统计标签的分布，了解数据中不同标签的数量。
	5.	保存处理结果：将处理后的数据保存为新的 CSV 文件，便于后续使用。
    通过这段代码，可以对原始日志数据进行预处理，确保数据的随机性，并生成一个新的子集供进一步的分析或模型训练使用。
'''

test = pd.read_csv('./robust_log_test.csv')
test = test.values

np.random.shuffle(test)
np.random.shuffle(test)
np.random.shuffle(test)
np.random.shuffle(test)

test = test[0:50000]
label = Counter(test[:,1])
print(label)
save_test = pd.DataFrame(data=test)
save_test.to_csv('./rubust_log_test_50000.csv',index=False,header=False)


