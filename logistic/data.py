from typing import Any, Tuple
import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

# MNIST dataset 
# train_data
class MnistTrainData(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.path = path
        self.train_data = torchvision.datasets.MNIST(root=self.path,
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index) ->Tuple[Any,Any]:
        return self.train_data[index]
# test_data
class MnistTestData(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.path = path
        self.test_data = torchvision.datasets.MNIST(root=self.path,
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, index) ->Tuple[Any,Any]:
        return self.test_data[index]
if __name__ == "__main__":
    data_path = '/media/g/学习资料盘/dataset/MNIST/data'
    train_data = MnistTrainData(data_path)
    print(len(train_data))
    test_data = MnistTestData(data_path)
    print(len(test_data))
