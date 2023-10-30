#!/usr/bin/python3.10
#coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, n_feature, layers, hidden_nodes):
        super(NeuralNet, self).__init__()

        assert(layers == len(hidden_nodes) + 1)

        input_node = n_feature

        self.module_list = nn.ModuleList()
        for  hidden_node in hidden_nodes:
            self.module_list.append(nn.Linear(input_node, hidden_node))
            #self.module_list.append(nn.Dropout(0.5))
            #self.module_list.append(nn.GELU())
            # self.module_list.append(nn.ReLU())
            self.module_list.append(nn.Tanh())
            # self.module_list.append(nn.LayerNorm(hidden_node))
            #self.module_list.append(nn.BatchNorm1d(hidden_node))
            input_node = hidden_node

        self.module_list.append(nn.Linear(input_node, 1))
        #self.module_list.append(nn.Tanh())


        for m in self.module_list:
            if isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight)
                #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.normal_(m.weight, mean=0, std=0.01)
                # nn.init.uniform_(m.weight, a=0, b=0.01)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class FM_model2(nn.Module):
    def __init__(self, n, k):
        super(FM_model2, self).__init__()
        self.n = n #len(items) + len(users)
        self.k = k
        self.nn = NeuralNet(n, 2, [15])
        #self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))
        self.act = nn.ReLU()

        nn.init.normal_(self.v, mean=0, std=0.01)

    def fm_layer(self, x):
        batch_size, slate_length, feature_dim = None, None, None
        if len(x.shape) == 3:
            batch_size, slate_length, feature_dim = x.shape
            x = x.reshape(-1, feature_dim)

        # x shape [batch, n]
        #linear_part = self.linear(x)
        linear_part = self.nn(x)

        # [batch, n] * [n, k]
        inter_part1 = torch.mm(x, self.v.t()) # out_size = [batch, k]
        # [batch, n]^2 * [n, k]^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = [batch, k]
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2, dim=1, keepdim=True)
        #output = linear_part + 0.5 * torch.mean(torch.pow(inter_part1, 2) - inter_part2)
        #print(linear_part)
        #print(0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2))
        #print(output)

        if batch_size != None and slate_length != None and feature_dim != None:
            output = output.reshape(batch_size, slate_length, -1)

        return output # out_size = [batch, 1]

    def forward(self, x):
        output = self.fm_layer(x)
        output = self.act(output)
        return output


class FM_model(nn.Module):
    def __init__(self, n, k):
        super(FM_model, self).__init__()
        self.n = n # len(items) + len(users)
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def fm_layer(self, x):
        # x 属于 R^{batch*n}
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)

        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        # 这里torch求和一定要用sum
        return output  # out_size = (batch, 1)

    def forward(self, x):
        output = self.fm_layer(x)
        return output

if __name__ == '__main__':
    #net = NeuralNet(100, 3 , [32, 15])
    net = NeuralNet(61, 4 , [64, 32, 15])
    print(net)

    # [batch_size, n_feature]
    x = torch.randn(32, 104, 61)
    print(x.shape)
    x = net(x)
    print(x.shape)


    x = torch.randn(32, 104, 61)
    fm = FM_model2(61, 4)
    x = fm(x)
    print(x.shape)

