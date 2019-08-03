import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import  model as md
import  dataprocess as DP
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import argparse
# TIME_STEP=50
# HIDDEN_SIZE=128
# TRAIN_RATE=0.7
# DATAPATH=u'E:/科研工作/Resource Allocation/DataMining/仿真/matlab/dataset2.mat'
# train_seq, test_seq = DP.DataGenerate(DATAPATH, TIME_STEP, TRAIN_RATE)# 取出数据len(train_seq[1][1])
# EPOCH=100
# OPIMIZER='Adam'
# INPUT_SIZE = 80

def model_training(paramters,train_seq):
    t0=time.time()

    #选择模型

    if paramters.model_mode  == 'LSTMpred':
        model = md.LSTMpred(paramters).cuda()
    elif  paramters.model_mode  == 'TrajPreSimple':
        model = md.TrajPreSimple(paramters).cuda()

    print('model:',model)

    # F.log_softmax(y)
    loss_function=nn.MSELoss()
    # loss_function =nn.NLLLoss()

    if paramters.optimizer=='Adam':
        optimizer = optim.Adam(model.parameters(), lr=paramters.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=paramters.learning_rate)

    #开始训练
    print('Start training...')
    count=0
    for epoch in range(paramters.epoch):
        for seq, outs in train_seq:
            count=count+1
            seq = torch.FloatTensor(seq).cuda()
            outs = torch.FloatTensor(outs).cuda()
            # print('seq.shape=',seq.shape)
            # print('outs.shape=', outs.shape)

            outs=torch.unsqueeze(outs,1)
            # outs = torch.from_numpy(np.array([outs]))

            optimizer.zero_grad()

            # model.hidden = model.init_hidden()

            modout = model(seq)


            loss = loss_function(modout, outs)
            loss.backward()
            optimizer.step()
            
#             if count%1000==0:
#                print( 'Epoch:', epoch, 'Loss:', loss.cpu().data.numpy())


        tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(tim1,'Epoch:',epoch,'Loss:',loss.cpu().data.numpy())

    t1 = time.time() - t0
    print('Training time: ', t1)
    torch.save(model,'model_rnn.pkl')

def model_test(test_seq):
    model=torch.load('model_rnn.pkl')
    predDat = []

    total=0
    for seq, trueVal in test_seq:
        seq = torch.FloatTensor(seq).cuda()
        trueVal = torch.FloatTensor(trueVal).cuda()

        predDat = model(seq)

        print('predDat=',predDat)
        print('trueVal=',trueVal)
        # total=total+1
        # if predDat:
    #     predDat.append(model(seq)[-1].data.numpy()[0])
    # fig = plt.figure()
    # plt.title(OPIMIZER)
    # plt.plot(y)
    # plt.plot(predDat)
    # plt.show()

if __name__ == '__main__':
    '''参数设计'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=80, help="input size")
    parser.add_argument('--hidden_size', type=int, default=500, help="hidden size")
    parser.add_argument('--output_size', type=int, default=80, help="output size")
    parser.add_argument('--time_step', type=int, default=96, help="time step")
    parser.add_argument('--data_path', type=str,
                        default=u'dataset2.mat', help="data path")
    parser.add_argument('--epoch', type=int, default=10, help="epoch size")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--traning_data_rate', type=int, default=0.7, help="traning_data_rate")
    parser.add_argument('--learning_rate', type=int, default=5 * 1e-4, help="leanning_rate")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--dropout_p', type=int, default=0.3, help="dropout_rate")
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--use_cuda', type=bool, default=True, choices=[True, False])
    parser.add_argument('--model_mode', type=str, default='LSTMpred',
                        choices=['LSTMpred', 'TrajPreSimple', 'attn_avg_long_user', 'attn_local_long'])

    paramters = parser.parse_args(['--input_size', '80'])

    np.random.seed(2)
    torch.manual_seed(2)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''取数据'''
    train_seq, test_seq = DP.DataGenerate(paramters)  # 取出数据len(train_seq[1][1])

    '''训练'''
    model_training(paramters,train_seq)

    '''测试'''
    # model_test(test_seq)
