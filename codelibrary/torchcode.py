# coding:utf-8
# some universal programs
import  torch

def gpuequipmet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device