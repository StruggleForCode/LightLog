# LightLog

## Introduction

LightLog 是一种基于开源深度学习的轻量级日志分析工具，用于日志异常检测。
## Function description
[BGL&HDFS dataset and Methods of data processing] is for the processing of time-series data
- BGL 包含从结构化日志构建 word2vec 模型的完整步骤
- Template saved as a .josn file
- 尽管在我们即将发表的论文中很少提及，但时间序列数据的处理与其他工作有很大不同。
  实验结果证明，该方法性能非常好
- [Enhanced TCN for Log Anomaly Detection on the BGL Dataset]
在 BGL 数据集上验证我们的方法
- [Enhanced TCN for Log Anomaly Detection on the HDFS Dataset]
在 HDFS 数据集上验证我们的方法

##Note: 
1. 这项工作包括 BGL、HDFS 数据集的处理、模型的训练和测试，包括构建 word2vec 模板、PCA-PPA 降维过程和改进 TCN 的细节 
2. This work does not include log parsing，if you need to use it, please check [logparser](https://github.com/logpai/logparser)*
3. 我们强烈推荐一些其他开源工作，作为我们工作的补充和比较。[logdeep]（https://github.com/donglee-afar/logdeep）
## Requirement

- python = 3.6
- tensorflow = 1.8.0
- Keras = 2.1.6
  
## Dataset
*Note: 
1. 我们的项目中提供了可用实验的数据。
2. 原始数据集太大，我们无法上传。因此，我们分享了一些链接供您下载。
   [BGL](https://www.kaggle.com/omduggineni/loghub-bgl-log-data) and Special thanks for this research work
   [HDFS](https://github.com/donglee-afar/logdeep/tree/master/data/hdfs)
   
