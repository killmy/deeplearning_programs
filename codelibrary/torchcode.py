# coding:utf-8
# some universal programs
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