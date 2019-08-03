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
import model as md
import dataprocess as DP
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import argparse
import json
from json import encoder
import sys

if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf8')
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def model_training(paramters):
    '''取数据'''
    train_seq, test_seq = DP.DataGenerate(paramters)  # 取出数据len(train_seq[1][1])

    t0 = time.time()

    '''选择模型'''
    if paramters.model_mode == 'LSTMpred':
        if paramters.pre_train == False:
            if paramters.use_cuda==True:
                model = md.LSTMpred(paramters).cuda()
            elif paramters.use_cuda==False:
                model = md.LSTMpred(paramters)

        elif paramters.pre_train == True:
            model = torch.load(paramters.model_mode + '_model.pkl')

    elif paramters.model_mode == 'TrajPreSimple':
        model = md.TrajPreSimple(paramters).cuda()
    print('model:', model)

    '''损失函数'''
    loss_function = nn.MSELoss()
    # loss_function =nn.NLLLoss()

    '''优化器'''
    if paramters.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=paramters.learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=paramters.learning_rate)

    metrics = {'time': [], 'epoch': [], 'Loss': [], 'test_avg_loss': []}

    '''开始训练'''
    print('Start training...')
    count = 0
    for epoch in range(paramters.epoch):
        for seq, outs in train_seq:
            count = count + 1
            if paramters.use_cuda==True:
                seq = torch.FloatTensor(seq).cuda()
                outs = torch.FloatTensor(outs).cuda()
            elif paramters.use_cuda==False:
                seq = torch.FloatTensor(seq)
                outs = torch.FloatTensor(outs)
            # seq = torch.FloatTensor(seq).cuda()
            # outs = torch.FloatTensor(outs).cuda()
            outs = torch.unsqueeze(outs, 1)

            modout = model(seq)
            loss = loss_function(modout, outs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每一百步保存一次模型
            if count % 500 == 0:
                torch.save(model, paramters.model_mode + '_model.pkl')
                print('Epoch:', epoch, 'Loss:', loss.cpu().data.numpy())

        test_avg_loss = model_test(test_seq)

        tim1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        metrics['time'].append(tim1)
        metrics['epoch'].append(int(epoch))
        metrics['Loss'].append(float(loss.cpu().data.numpy()))
        metrics['test_avg_loss'].append(float(test_avg_loss))

        print(tim1, '  Epoch:', epoch, '  Loss:', loss.cpu().data.numpy(), '  Test_avg_loss:', test_avg_loss)

    t1 = time.time() - t0
    print('Total training time: ', t1)
    torch.save(model, paramters.model_mode + '_model.pkl')
    json.dump({'metrics': metrics}, fp=open('Training_process' + tim1 + '.rs', 'w'), indent=4)


def model_test(test_seq):
    model = torch.load(paramters.model_mode + '_model.pkl')

    loss_function = nn.MSELoss()

    predDat = []
    total = 0
    loss_total = []

    for seq, trueVal in test_seq:
        if paramters.use_cuda == True:
            seq = torch.FloatTensor(seq).cuda()
            trueVal = torch.FloatTensor(trueVal).cuda()
        elif paramters.use_cuda == False:
            seq = torch.FloatTensor(seq)
            trueVal = torch.FloatTensor(trueVal)
        # seq = torch.FloatTensor(seq).cuda()
        # trueVal = torch.FloatTensor(trueVal).cuda()

        modout = model(seq)
        loss = loss_function(modout, trueVal)

        loss_total.append(loss.cpu().data.numpy())

    #     print('Test_avg_loss',sum(loss_total) / len(loss_total))

    return sum(loss_total) / len(loss_total)


#         predDat.append(model(seq)[-1].data.cpu().numpy()[0])
#     fig = plt.figure()
#     plt.title('Adam')
#     plt.plot(dataset)

#     plt.plot(predDat)
#     plt.show()

####简单sin(x)回归问题
#     train_seq, test_seq, dataset= DP.Generate_data_demo(paramters)  # 取出数据len(train_seq[1][1])


if __name__ == '__main__':
    '''参数设计'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=512, help="hidden size")
    parser.add_argument('--time_step', type=int, default=96, help="time step")
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--traning_data_rate', type=int, default=0.8, help="traning_data_rate")
    parser.add_argument('--learning_rate', type=int, default=2 * 1e-3, help="leanning_rate")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--dropout_p', type=int, default=0.4, help="dropout_rate")
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'])
    parser.add_argument('--use_cuda', type=bool, default=False, choices=[True, False])
    parser.add_argument('--pre_train', type=bool, default=False, choices=[True, False])
    parser.add_argument('--nomalization', type=bool, default=True, choices=[True, False])
    parser.add_argument('--model_mode', type=str, default='LSTMpred',
                        choices=['LSTMpred', 'TrajPreSimple', 'attn_avg_long_user', 'attn_local_long'])

    parser.add_argument('--input_size', type=int, default=80, help="input size")
    parser.add_argument('--output_size', type=int, default=80, help="output size")

    '''参数'''
    parser.add_argument('--data_path', type=str, default=u'dataset3.mat', help="data path")
    parser.add_argument('--epoch', type=int, default=200, help="epoch size")

    paramters = parser.parse_args(['--input_size', '80'])

    np.random.seed(3)
    torch.manual_seed(3)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''训练'''
    model_training(paramters)

#     '''测试'''
#     train_seq, test_seq = DP.DataGenerate(paramters)  # 取出数据len(train_seq[1][1])
#     model_test(test_seq)
