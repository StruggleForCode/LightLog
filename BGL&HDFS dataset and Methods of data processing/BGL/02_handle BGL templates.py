import numpy as np
from tensorflow.keras.layers import Input,Embedding,Lambda
from tensorflow.keras.models import Model,load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow.keras.backend as K
import pandas as pd
import json

'''
    这份代码通过以下几个主要步骤实现了 word2vec 词向量的训练与句向量的生成：
	•	数据准备：从 CSV 文件中读取模板文本，并分词。
	•	字典构建与降采样：统计词频、生成词典，并根据词频计算降采样概率，以减少高频词对训练的影响。
	•	训练数据生成：以 CBOW 模型的方式构造训练样本，每个样本利用上下文词预测中心词，并利用负采样生成负样本。
	•	模型构建：使用 Keras 搭建 CBOW 模型，其中利用 Embedding 层保存词向量，通过负采样构造 softmax 层只在一小部分采样内计算概率分布。
	•	训练与保存：利用生成的数据训练模型，并将训练好的词向量提取、归一化，最后对每个句子将其词向量相加得到句向量并保存为 JSON。
    这种实现方法虽然简化了 word2vec 的细节，但基本体现了 CBOW 模型与负采样的核心思想。希望这个详细解释能帮助你更好地理解代码的每一步操作及其背后的原理。
'''

word_size = 300  # 词向量维度
window = 5  # 窗口大小
nb_negative = 15  # 随机负采样的样本数
min_count = 0  # 频数少于min_count的词将会被抛弃
nb_worker = 1  # 读取数据的并发数
nb_epoch = 20  # 迭代次数，由于使用了adam，迭代次数1～2次效果就相当不错
subsample_t = 1e-5  # 词频大于subsample_t的词语，会被降采样，这是提高速度和词向量质量的有效方案
nb_sentence_per_batch = 30  # 目前是以句子为单位作为batch，多少个句子作为一个batch（这样才容易估计训练过程中的steps参数，另外注意，样本数是正比于字数的。）



def getdata():
    data = pd.read_csv('./bgl/templates.csv').values
    templates = data[:,1]
    label = data[:,0]
    sentences = []
    for s in templates:
        sentences.append(s.split())
    return label,templates, sentences



def bulid_dic(sentences):  # 建立各种字典
    words = {}       # 用来统计每个词的频数
    nb_sentence = 0  # 总句子数
    total = 0.       # 总词数（累计所有词的出现次数）

    for d in sentences:
        nb_sentence += 1
        for w in d:
            if w not in words:
                words[w] = 0
            words[w] += 1
            total += 1
        if nb_sentence % 100 == 0:
            pass  # 可以在此处添加日志输出

    # 舍弃频数小于 min_count 的词（这里 min_count 为0，所以实际上不会舍弃）
    words = {i: j for i, j in words.items() if j >= min_count}

    # 构造 id 到词的映射，从 1 开始编号，0 留作填充或未知词
    id2word = {i + 1: j for i, j in enumerate(words)}
    # 构造词到 id 的映射
    word2id = {j: i for i, j in id2word.items()}
    # 总词数，加上 0 对应的填充符号
    nb_word = len(words) + 1

    # 计算每个词的相对频率（仅对频率高于阈值的词进行降采样）
    subsamples = {i: j / total for i, j in words.items() if j / total > subsample_t}
    # 按照 word2vec 中的降采样公式计算概率：公式为 t/f + sqrt(t/f)
    subsamples = {i: subsample_t / j + (subsample_t / j) ** 0.5 for i, j in subsamples.items()}
    # 将键由词变为对应的 id，并过滤掉那些概率大于等于1（保留概率低于1的）
    subsamples = {word2id[i]: j for i, j in subsamples.items() if j < 1.}
    return nb_sentence, id2word, word2id, nb_word, subsamples


def data_generator(word2id, subsamples, data):  # 训练数据生成器
    x, y = [], []
    _ = 0
    for d in data:
        # 为了方便取上下文窗口，句子前后各补 window 个 0（填充符号）
        d = [0] * window + [word2id[w] for w in d if w in word2id] + [0] * window
        # 为句子中的每个位置生成一个随机数，用于降采样高频词
        r = np.random.random(len(d))
        # 遍历句子中每个非填充的位置
        for i in range(window, len(d) - window):
            # 如果当前词在降采样字典中，并且随机数大于降采样概率，则跳过（丢弃该词）
            if d[i] in subsamples and r[i] > subsamples[d[i]]:
                continue
            # 将当前位置的上下文（窗口内左右各 window 个词）作为输入
            x.append(d[i - window:i] + d[i + 1:i + 1 + window])
            # 当前词作为目标输出（注意用列表包裹，保持维度一致）
            y.append([d[i]])
        _ += 1
        # 当处理了 nb_sentence_per_batch 个句子后，返回一个 batch 数据
        if _ == nb_sentence_per_batch:
            x, y = np.array(x), np.array(y)
            # 注意：下面 z 全部为 0，因为在模型中我们构造采样时把正样本放在第一位，
            # 所以目标的下标始终是 0。z 就是这个标签。
            z = np.zeros((len(x), 1))
            return [x, y], z


def build_w2vm(word_size, window, nb_word, nb_negative):
    K.clear_session()  # 清除之前的模型，避免内存占用累积

    # --- 构造 CBOW 输入部分 ---
    # 输入为上下文词 id 序列，长度为 window*2（左右各 window 个词）
    input_words = Input(shape=(window * 2,), dtype='int32')
    # 使用 Embedding 层将词 id 转换成词向量，Embedding 层的权重即为词向量矩阵
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
    # 对上下文中所有词的向量求和，得到 CBOW 模型中的上下文表示
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)

    # --- 构造负采样部分 ---
    # 目标词输入，形状为 (1,) —— 当前需要预测的词
    target_word = Input(shape=(1,), dtype='int32')
    # 生成负样本：利用 Lambda 层生成形状为 (batch_size, nb_negative) 的随机整数张量，范围在 [0, nb_word)
    negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
    # 将正样本（目标词）与负样本拼接，注意正样本在第一位。构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。
    samples = Lambda(lambda x: K.concatenate(x))([target_word, negatives])

    # --- 在采样集合上计算得分 ---
    # 构造用于计算得分的 Embedding 层，名称 'W'，用于存储各词的权重向量
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    # 构造用于计算偏置的 Embedding 层，名称 'b'
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    # 利用 Lambda 层实现矩阵乘法：对每个采样词，计算上下文向量与采样词权重的点乘，再加上偏置
    # 用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能
    softmax = Lambda(lambda x:
                     K.softmax((K.batch_dot(x[0], K.expand_dims(x[1], 2)) + x[2])[:, :, 0])
                     )([softmax_weights, input_vecs_sum, softmax_biases])
    # 解释：
    # - softmax_weights 的 shape 为 (batch_size, nb_negative+1, word_size)
    # - input_vecs_sum 的 shape 为 (batch_size, word_size)，expand_dims 后为 (batch_size, word_size, 1)
    # - K.batch_dot 计算每个采样词与上下文向量的内积，结果 shape 为 (batch_size, nb_negative+1, 1)
    # - 加上 softmax_biases 后，再 squeeze 最后一维，得到 (batch_size, nb_negative+1)
    # - 最后对该维度做 softmax，输出各采样词的概率分布

    # 构造模型：输入为上下文和目标词，输出为经过 softmax 的概率分布
    # 留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出
    model = Model(inputs=[input_words, target_word], outputs=softmax)
    # 使用 sparse_categorical_crossentropy 作为损失函数，因为标签是整数（正样本总在位置 0）
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 请留意用的是sparse_categorical_crossentropy而不是categorical_crossentropy
    model.summary()
    return model


if __name__ == '__main__':
    # word2vector
    # 1. 加载数据，获得标签、原始文本和分词后的句子
    label, templates, sentences = getdata()

    # 2. 构建词典，得到句子总数、id<->word 映射、词表大小以及降采样字典
    nb_sentence, id2word, word2id, nb_word, subsamples = bulid_dic(sentences)

    # 3. 构造一个 batch 的训练数据（注意：这里传入的是 templates，要求每个模板已分词，
    #    否则应传入 sentences。代码中假设模板已经是分好词的列表。）
    ipt, opt = data_generator(word2id, subsamples, templates)

    # 4. 构建 CBOW 模型（带负采样）
    model = build_w2vm(word_size, window, nb_word, nb_negative)

    # 5. 训练模型
    # steps_per_epoch 为总句子数除以每个 batch 的句子数，epochs 为总迭代次数
    model.fit(ipt, opt, steps_per_epoch=int(nb_sentence / nb_sentence_per_batch), epochs=nb_epoch)

    # 6. 保存训练好的模型
    model.save('word2vec.h5')

    # 7. 从模型中提取词向量
    embeddings = model.get_weights()[0]  # 对应 Embedding 层 'word2vec' 的权重矩阵
    # 对词向量进行归一化，使每个词向量的 L2 范数为 1
    normalized_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5

    # 8. 生成句向量并保存为 JSON 文件
    vector_json = {}
    for i in range(0, len(sentences)):
        vector = []
        # 对于句子中的每个词，获取其归一化后的词向量
        for ii in sentences[i]:
            vector.append(normalized_embeddings[word2id[ii]])
        # 将句子中所有词向量求和作为该句的向量表示
        # (i+1) 作为 key，值转换为 list 后存储到字典中
        vector_json.update({(i + 1): list(np.float64(np.sum(vector, axis=0)))})
    # 转换为 JSON 字符串后写入文件
    json_str = json.dumps(vector_json)
    with open('./bgl/bgl_templates.json', 'w') as json_file:
        json_file.write(json_str)