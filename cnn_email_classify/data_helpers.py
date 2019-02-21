import numpy as np
import re
import itertools
from collections import Counter


# 清洗数据  返回小写
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels(positive_data_file, negative_data_file):
    positive = open(positive_data_file, 'rb').read().decode('utf-8')
    negative = open(negative_data_file, 'rb').read().decode('utf-8')

    positive_examples = positive.split('\n')[:-1]  # 倒序
    negative_examples = negative.split('\n')[:-1]  #

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # 生成lable
    positive_lables = [[0, 1] for _ in positive_examples]
    negative_lables = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_lables, negative_lables], 0)

    return [x_text, y]

def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data = np.array(data)
    data_size = len(data) #一共多少段的数据(小文章)，一个文章中有多个词
    num_batches_per_epoch = int((len(data)-1)/batch_size)+1 #一个epoch被拆成了多少个batch
    for epoch in range(num_epochs):
        # 打乱epoch中的数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.array(data_size)) #打乱顺序后的index
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        #
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num+1)*batch_size,data_size)
            yield shuffle_data[start_index:end_index]
            # yield 生成generator对象， 遵循迭代器（iterator）协议，迭代器协议需要实现__iter__、next接口
            # 能过多次进入、多次返回，能够暂停函数体中代码的执行

















