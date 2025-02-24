import matplotlib.pyplot as plt
import re

# 读取文件内容
# file_path = "BGL_DIM30_EPOCH150.txt"
#file_path = "train_pca_dim30_epoch150_output1.log"
# file_path = "train_pca_dim20_epoch150_output.log"
#file_path = "train_ppa_dim30_epoch150_output3.log"
# file_path = "Train_HDFS_PCA_Dim20"
file_path= "smoothed_train_pca_dim30_epoch150_output.log"
#file_path = "test.txt"
with open(file_path, "r") as file:
    lines = file.readlines()

# 初始化数据存储
epochs = []
train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []

# 正则表达式匹配数值
pattern = re.compile(
    r"accuracy: (\d+\.\d+) - loss: (\d+\.\d+) - val_accuracy: (\d+\.\d+) - val_loss: (\d+\.\d+)"
)

epoch_counter = 0
# 解析数据
for i, line in enumerate(lines):
    match = pattern.search(line)
    if match:
        epoch_counter += 1
        train_acc, train_ls, val_acc, val_ls = map(float, match.groups())
        epochs.append(epoch_counter)
        train_accuracy.append(train_acc)
        train_loss.append(train_ls)
        val_accuracy.append(val_acc)
        val_loss.append(val_ls)

# 绘制 accuracy 图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label="train", linestyle="-")
plt.plot(epochs, val_accuracy, label="test", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("TCN model accuracy in BGL")
plt.legend()
plt.show()

# 绘制 loss 图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="train", linestyle="-")
plt.plot(epochs, val_loss, label="test", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("TCN model loss in BGL")
plt.legend()
plt.show()
