# codeing:utf-8 
# logistic model
import torch
import torch.nn as nn
class LogisticModel(nn.Module):
    def __init__(self,input_size,class_nums):
        super(LogisticModel,self).__init__()
        self.fc = nn.Linear(input_size,class_nums)
    
    def forward(self,x):
        out = self.fc(x)
        return out

if __name__ == "__main__":
    import sys
    sys.path.append("/media/g/学习资料盘/deeplearning_programs/")
    import Visualization.visualization as v
    input_size = 28*28
    class_nums = 10
    model = LogisticModel(input_size,class_nums)
    print(model)
    data = torch.rand([1,input_size])
    path = './logistic/picture/model1'
    theme = "blue"
    view = v.HVisualizationModel(model=model,data=data,path=path,theme=theme)
    view.save()
    path1 = './logistic/picture/'
    font = 'png'
    view = v.TVisualizationModel(model,data,font,path1)
    view.save()
