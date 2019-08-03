import torch
import torch.nn as nn
from torch.autograd import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from torch.autograd import Variable

class LSTMpred(nn.Module):
    def __init__(self, paramters):
        super(LSTMpred, self).__init__()

        self.time_step=paramters.time_step
        self.batch_size=paramters.batch_size
        self.input_size=paramters.input_size

        self.hidden_size = paramters.hidden_size

        self.output_size = paramters.output_size
        
        self.use_cuda=paramters.use_cuda
        
        self.lstm = nn.LSTM(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=paramters.dropout_p)
        
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:

            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, x):
        
        x=x.view(self.time_step, self.batch_size,-1)


        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        
        if self.use_cuda==True:
            h1=h1.cuda()
            c1=c1.cuda()

        out, (h1, c1) = self.lstm(x, (h1, c1))

        # print('out.shape1=', out.shape)

        out = out.squeeze(1)  #
        out = F.selu(out)  # F是nn公式
        out = self.dropout(out)
        out = self.fc(out)

        # print('out.shape=',out.shape)
#         print('out.shape=', out[-1, :].shape)

        out=out[-1, :] #取最后的一个时间点输出
        out = out.view(-1,self.output_size)
        
        return out



        # out, (h1, c1) = self.lstm(
        #     seq.view(len(seq), 1, -1))

        # print('seq.view(len(seq), 1, -1)', seq.view(len(seq), 1, -1).shape)
        # print('lstm_out', lstm_out.shape)

        # outdat = self.fc(lstm_out.view(len(seq), -1))

        # print('lstm_out.view(len(seq), -1)',lstm_out.view(len(seq), -1).shape)
        # return out[-1,:]

# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model""" #基准模型

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        # self.loc_size = parameters.loc_size
        # self.loc_emb_size = parameters.loc_emb_size
        # self.tim_size = parameters.tim_size
        # self.tim_emb_size = parameters.tim_emb_size

        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type
        self.input_size = parameters.input_size
        self.hidden_size = parameters.hidden_size
        self.output_size = paramters.output_size

        self.rnn = nn.LSTM(input_size, self.hidden_size, 1)


        # self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        # self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        #
        # input_size = self.loc_emb_size + self.tim_emb_size

        # if self.rnn_type == 'GRU':
        #     self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        # elif self.rnn_type == 'LSTM':
        #     self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        # elif self.rnn_type == 'RNN':
        #     self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        # self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, x):
        h1 = Variable(torch.zeros(1, 1, self.hidden_size))
        c1 = Variable(torch.zeros(1, 1, self.hidden_size))
        # if self.use_cuda:
        #     h1 = h1.cuda()
        #     c1 = c1.cuda()
        # print('======================================================================')
        # print('loc_size=',self.loc_size)
        # print('loc_emb_size=', self.loc_emb_size)
        # loc_emb = self.emb_loc(loc)
        # tim_emb = self.emb_tim(tim)

        # x = torch.cat((loc_emb, tim_emb), 2)

        # print('loc=', loc.shape)
        # print('tim', tim.shape)
        #
        # print('loc_emb.shape=',loc_emb.shape)
        # print('tim_emb.shape=', tim_emb.shape)



        # print('x.shape=',x.shape)


        x = self.dropout(x)

        print('x.dropout=', x.shape)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out = out.squeeze(1) #
        print(out.shape)
        out = F.selu(out) #F是nn公式
        out = self.dropout(out)

        y = self.fc(out)


        score = F.log_softmax(y)  # calculate loss by NLLoss
        return score