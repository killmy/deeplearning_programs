# coding:utf-8
# some universal programs
from typing import Any, Tuple
import  torch
import os
# 加载设备单GPU
def gpuequipmet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# 加载模型
class LoadModel():
    def __init__(self) -> None:
        pass
    def add_params(self,model,path):
        'if path is wrong,it will return  origin model'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path))
            return model
        else:
            return model
    def add_model(self,path,device):
        'if path is wrong,it will return 0'
        if os.path.exists(path):
            model = torch.load(path).to(device)
            return model
        else:
            print('no model can be loaded,please check the path')
            return 0

# 分类准确率预测
class ClassAcc():
    def __init__(self) -> None:
        pass
    def predict(self,model,loader,device)->Tuple[Any,Any]:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images,labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _,predicted = torch.max(outputs,1)
                total +=labels.size(0)
                correct += (predicted == labels).sum().item()
        # model.train()
        return correct,total
class ClassAccTwo():
    def __init__(self) -> None:
        pass
    def predict(self,model,images,labels,device)->Tuple[Any,Any]:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
            total +=labels.size(0)
            correct += (predicted == labels).sum().item()
        # model.train()
        return correct,total

class classPredict():
    def __init__(self) -> None:
        pass
    def predict(self,model,images,device)->Any:
        model.eval()
        with torch.no_grad():
            images = images.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs,1)
        return predicted

