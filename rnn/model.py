import torch
import torch.nn as nn
from torch.autograd import Variable

class seqRNN(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, num_layer):
        super(seqRNN, self).__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.num_layer = num_layer
        # self.model = 'lstm' # hardcoded LSTM model

        self.encoder = nn.Embedding(num_in, num_hidden)
        # self.rnn = nn.LSTM(num_hidden, num_hidden, num_layer)
        self.rnn = nn.GRU(num_hidden, num_hidden, num_layer)
        self.decoder = nn.Linear(num_hidden, num_out)


    def forward(self, input, hidden):
        num_batch = input.size(0)
        if num_batch > 1:
            encode   =self.encoder(input)
            output, hidden = self.rnn(encode.view(1, num_batch, -1), hidden)
            output = self.decoder(output.view(num_batch, -1))
        else:
            encode = self.encoder(input.view(1,-1))
            output, hidden  = self.rnn(encode.view(1,1,-1),hidden)
            output = self.decoder(output.view(1,-1))
        return output, hidden




    def init_hidden(self, num_batch):
        # return (Variable(torch.zeros(self.num_layer, num_batch, self.num_hidden)),
        #         Variable(torch.zeros(self.num_layer, num_batch, self.num_hidden)))

        return Variable(torch.zeros(self.num_layer, num_batch, self.num_hidden))
