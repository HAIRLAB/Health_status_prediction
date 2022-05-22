import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut, dropout):
        super(BidirectionalLSTM, self).__init__()
        """
        Args:
            nIn (int): The number of input unit
            nHidden (int): The number of hidden unit
            nOut (int): The number of output unit
        """
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=False, batch_first=True)
        self.embedding = nn.Linear(nHidden, nOut)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = dropout

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(b * T, h)
        
        if self.dropout:
            t_rec = self.dropout(t_rec)
        output = self.embedding(t_rec)
        output = output.contiguous().view(b, T, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, ni, nc, no, nh, n_rnn=2, leakyRelu=False,sigmoid = False):
        """
        Args:
            ni (int): The number of input unit
            nc (int): The number of original channel
            no (int): The number of output unit
            nh (int): The number of hidden unit
        """
        super(CRNN, self).__init__()

        ks = [3, 3, 3,    3, 3,   3, 3]
        ps = [0, 0, 0,    0, 0,   0, 0]
        ss = [2, 2, 2,    2, 2,   2, 1]
        nm = [8, 16, 64,  64, 64, 64, 64]

        cnn = nn.Sequential()

        def convRelu(i, cnn, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            if i == 3: nIn = 64
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, (ks[i],1), (ss[i],1), (ps[i],0)))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0,cnn)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2,1), (2,1)))
        convRelu(1,cnn)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2,1), (2,1)))
        convRelu(2,cnn)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 1), (2, 1), (0, 0)))
        self.sigmoid = sigmoid
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(64, nh, nh, False),
            BidirectionalLSTM(nh, nh, no, False),)
        self.rul = nn.Linear(10, 1)
        self.soh = nn.Linear(64, 1)


    def forward(self, input):
        """
        Input shape: [b, c, h, w]
        Output shape: 
            rul [b, 1]
            soh [b, 10]
        """
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        output = self.rnn(conv)
        soh = self.soh(output).squeeze()
        
        if not self.sigmoid:
            rul = self.rul(soh)
        else:
            rul = F.sigmoid(self.rul(soh))
    
        return rul, soh
        