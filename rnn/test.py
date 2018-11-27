import torch
import os
import string 

from model import *

len_init = 100

def char2Tensor(str):
    rval = torch.zeros(len(str)).long()
    for i in range(len(str)):
        rval[i] = string.printable.index(str[i])
    return rval


file = open('constitute.txt').read()
len_file = len(file)

def main():
    net = torch.load('federalist.pt')
    net.cuda()

    hidden = net.init_hidden(1)
    # hidden = (hidden[0].cuda(), hidden[1].cuda())
    hidden = hidden.cuda()
    count = 0
    hitcount = 0

    for k in range(0,10000,10):
        init_parag = file[k:k+100]
        y_parag = init_parag
        z_parag = file[k:k+110]
        init_input = char2Tensor(init_parag).unsqueeze(0)
        init_input = Variable(init_input).cuda()

        for i in range(len(init_parag)):
            _, hidden = net(init_input[:,i], hidden)

        x = init_input[:,-1]

        for i in range(1):
            output, hidden = net(x, hidden)

            dist = output.data.view(-1).div(0.8).exp()
            idx = torch.multinomial(dist,1)[0]

            nextChar = string.printable[idx]
            y_parag += nextChar;
            x = Variable(char2Tensor(nextChar).unsqueeze(0)).cuda()


        count += 1
        if z_parag[100]==y_parag[100]:
            hitcount += 1
        
        # print('==========')
        # print(init_parag)
        # print('--')
        # print(z_parag)
        # print('--')
        # print(y_parag)
        # print('==========')

    print(hitcount)
    print(count)


if __name__ == '__main__':
    main()
