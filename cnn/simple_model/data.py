""" 
MNIST数据集
"""
import torch 
import torchvision
from torchvision.transforms import transforms

# data 地址
data_path = '/media/g/学习资料盘/dataset/MNIST/data'

# MNIST DATA
train_dataset = torchvision.datasets.MNIST(root=data_path,
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=data_path,
                                          train=False, 
                                          transform=transforms.ToTensor())

if __name__ == "__main__":
    print(train_dataset[0])