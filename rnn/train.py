import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import string
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import *
import numpy as np
import random

num_epoch = 2000
num_hidden = 100
num_layer = 2
lr = 0.01
len_parag = 200
num_batch = 100


file = open('federalist.txt').read()
len_file = len(file)


def char2Tensor(str):
    rval = torch.zeros(len(str)).long()
    for i in range(len(str)):
        rval[i] = string.printable.index(str[i])
    return rval


def char2ndarray(str):
    rval = np.zeros(len(str),dtype=np.int64)
    for i in range(len(str)):
        rval[i] = string.printable.index(str[i])
    return rval


class DatasetHolder(Dataset):

    def __init__(self, filename, transform):
        self.file = open(filename).read()
        self.len_file = len(self.file)
        self.transform = transform


    def __len__(self):
        return self.len_file


    def __getitem__(self, i):
        endIdx = i + len_parag + 1
        parag = self.file[i:endIdx]
        x = char2ndarray(parag[:-1])
        z = char2ndarray(parag[1:])
        x = self.transform(x)
        x = Variable(x)
        z = Variable(z)
        return {'x': x, 'z': z}


def train_set(len_parag, num_batch):
    x = torch.LongTensor(num_batch, len_parag)
    z = torch.LongTensor(num_batch, len_parag)
    for i in range(num_batch):
        si = random.randint(0, len_file - len_parag)
        ei  = si + 1 + len_parag
        parag = file[si:ei]
        x[i] = char2Tensor(parag[:-1])
        z[i] = char2Tensor(parag[1:])
    return {'x': x, 'z':z}



def main():
    ## train_set = DatasetHolder('federalist.txt',transform=transforms.ToTensor())
    ## train_load= DataLoader(train_set, batch_size = num_batch, num_workers=1)

    ## seqRNN(nn.Module):
        ## def __init__(self, num_in, num_hidden, num_out, num_layer):
    net = seqRNN(len(string.printable), num_hidden, len(string.printable), num_layer)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    ## net = torch.nn.DataParallel(net)
    net.cuda()

    for epoch in range(num_epoch):
        data = train_set(len_parag,num_batch)
        x = Variable(data['x']).cuda()
        z = Variable(data['z']).cuda()

        hidden = net.init_hidden(num_batch)
        # hidden = (hidden[0].cuda(), hidden[1].cuda())
        hidden = hidden.cuda()
        net.zero_grad()
        loss = 0

        for j in range(len_parag):
            output, hidden = net(x[:,j], hidden)
            output = output.view(num_batch,-1)
            loss += criterion(output,z[:,j])

        loss.backward()
        optimizer.step()

    torch.save(net, 'federalist.pt')

if __name__ == "__main__":
    main()
