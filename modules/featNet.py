#!/usr/bin/env python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.cbam import *
#from cbam import *


class Basic_Block(nn.Module):
    def __init__(self,In,Out,ks,std,pad):
        super(Basic_Block,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(In,Out,ks,std,pad),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(Out,Out,ks,std,pad),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self,x):
        return self.conv(x)


class Attention_Block(nn.Module):
    def __init__(self,In,Out,ks,std,pad):
        super(Attention_Block,self).__init__()
        self.bb = Basic_Block(In,Out,ks,std,pad)
        self.cbam = CBAM(Out,16)

    def forward(self,x):
        out = self.bb(x)
        out = self.cbam(out)
        return out


def ShuffleDown(x,r):
    device = x.device
    B,C,H,W = x.size()
    out = torch.zeros((B,int(C*r**2),\
            int(H/r),int(W/r))).to(device)
    k = 0
    for i in range(r):
        for j in range(r):
            out[:,k::r**2,:,:] = x[:,:,i::r,j::r]
            k += 1
    return out


class ConvRelu(nn.Module):
    def __init__(self,In,Out,ks=3,std=1,pad=1):
        super(ConvRelu,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(In,Out,ks,std,pad),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self,x):
        return self.conv(x)


class ResidualBlock3(nn.Module):
    def __init__(self,In,ks=3,std=1,pad=1):
        super(ResidualBlock3,self).__init__()
        self.conv = nn.Sequential(
                ConvRelu(In,In),
                ConvRelu(In,In),
                ConvRelu(In,In),
                nn.Conv2d(In,In,ks,std,pad)
            )

    def forward(self,x):
        return self.conv(x)+x


class TopNet(nn.Module):
    def __init__(self,In,Out,ks=3,std=1,pad=1):
        super(TopNet,self).__init__()
        self.conv = nn.Sequential(
                ConvRelu(In,Out),
                ResidualBlock3(Out),
                ConvRelu(Out,Out)
            )

    def forward(self,x):
        return F.pixel_shuffle(self.conv(x),2)


class MidNet(nn.Module):
    def __init__(self,In,Out,Outp,ks=3,std=1,pad=1):
        super(MidNet,self).__init__()
        self.conv1 = ConvRelu(In,Out)
        self.conv2 = nn.Sequential(
                ConvRelu((Out+Outp),Out),
                ResidualBlock3(Out),
                ConvRelu(Out,Out)
            )

    def forward(self,x,xp):
        out = self.conv1(x)
        out = self.conv2(torch.cat((out,xp),1))
        return F.pixel_shuffle(out,2)


class BotNet(nn.Module):
    def __init__(self,In,Out,Outp,Outf,ks=3,std=1,pad=1):
        super(BotNet,self).__init__()
        self.conv1 = ConvRelu(In,Out)
        self.conv2 = nn.Sequential(
                ConvRelu((Out+Outp),Out),
                ConvRelu(Out,Out),
                ConvRelu(Out,Out),
                nn.Conv2d(Out,Outf,ks,std,pad)
            )

    def forward(self,x,xp):
        out = self.conv1(x)
        out = self.conv2(torch.cat((out,xp),1))
        return out


class SelfGuidedNet(nn.Module):
    def __init__(self,In,Out,C,K):
        super(SelfGuidedNet,self).__init__()
        topC = 2**K
        self.top = TopNet(In*4**K,C*topC)
        self.mid = nn.ModuleList()
        prev = topC*C//4
        for i in range(K-1,0,-1):
            midC = 2**i
            self.mid.append(MidNet(In*4**i,C*midC,prev))
            prev = midC*C//4
        self.bot = BotNet(In,C,prev,Out)
        self.K = K

    def forward(self,x):
        out = ShuffleDown(x,2**self.K)
        out = self.top(out)
        for i in range(self.K-1):
            tmp = ShuffleDown(x,2**(self.K-1-i))
            out = self.mid[i](tmp,out)
        return self.bot(x,out)


class Out_Branches(nn.Module):
    def __init__(self,In,Ks,N):
        super(Out_Branches,self).__init__()
        self.bb = Basic_Block(In,In,3,1,1)
        abOut = (Ks**2+1)*N
        self.ab = Attention_Block(In,In,3,1,1)
        self.bhk = nn.Conv2d(In,abOut,1,1)
        self.bhw = nn.Conv2d(In,N,1,1)

    def forward(self,x):
        kr = self.ab(x)
        kr = self.bhk(kr)
        W = self.bb(x)
        W = torch.sigmoid(self.bhw(W))
        return kr,W



if __name__ == '__main__':
    import os
    import time
    os.environ["CUDA_VISIBLE_DEVICES"]='2'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1,1,128,128).to(device)
    net = SelfGuidedNet(1,1,32,3)
    net.to(device)
    net.eval()
    import pdb;pdb.set_trace()
    t0 = time.time()
    ofeat = net(x)
    t1 = time.time()
    print('cost %f'%((t1-t0)))
    print(ofeat.shape)
