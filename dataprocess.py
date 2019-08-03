import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import numpy as np


def SeriesGen(N):
    x = torch.arange(1, N, 0.01)
    return torch.sin(x)   #+torch.cos(x)+torch.pow(x,2), x 

def trainDataGen(seq, k):
    dat = list()
    L = len(seq)
    for i in range(L - k - 1):
        indat = seq[i:i + k]
        outdat = seq[i + k:i + k + 1]
        dat.append((indat, outdat))
    return dat

'''数据归一化到0-1'''
def normalization(y):
    max_value = np.max(y)
    min_value = np.min(y)
    scalar = max_value - min_value
    y = list(map(lambda x: x / scalar, y))
    return  y

'''产生数据'''
# matlab文件名

def DataGenerate(paramters):

    matfn = paramters.data_path
    data = sio.loadmat(matfn)
    # print(data['dataset'])

    #取出数据
    list=data['dataset']
    #将数据转化为numpy
    data=np.array(list)
    
    if paramters.nomalization==True:
        data=normalization(data)

    # print('data=',type(data))
    # print(data[1:3])

    data = trainDataGen(data, paramters.time_step)

    '''训练数据和测试数据设置'''
    train_size = int(len(data) * paramters.traning_data_rate)
    test_size = len(data) - train_size

    train_seq = data[:train_size]
    test_seq = data[train_size:]
    
    return  train_seq, test_seq

def Generate_data_demo(paramters):
    
    dataset=SeriesGen(20).numpy()
    
    data=normalization(dataset)

    # print('data=',type(data))
    # print(data[1:3])

    data = trainDataGen(data, paramters.time_step)

    '''训练数据和测试数据设置'''
    train_size = int(len(data) * paramters.traning_data_rate)
    test_size = len(data) - train_size

    train_seq = data[:train_size]
    test_seq = data[train_size:]

    return  train_seq, test_seq, dataset  

