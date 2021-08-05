import torch
import torch.nn as nn
from torch.nn.modules import module

class LinearModel(nn.Module):
    def __init__(self):
        # super(LinearModel,self).__init__() 继承父类的属性与方法
        # 不加入强继承是没有办法继承属性的
        super(LinearModel,self).__init__()
        self.fc = nn.Linear(1,1)
    def forward(self,x):
        out = self.fc(x)
        return out
if __name__ == "__main__":
    import sys
    # sys.path.append('E:/deeplearning_programs/')
    import data as d
    import Visualization.visualization as v
    model  = LinearModel()
    print(model)
    train_data = d.TrainData(d.x_train,d.y_train)
    train_data_x,train_data_y = train_data[0]
    train_data_x = torch.unsqueeze(train_data_x,dim=0)
    path = './linear/picture/model1'
    theme = "blue"
    view = v.HVisualizationModel(model=model,data=train_data_x,path=path,theme=theme)
    view.save()
    path1 = './linear/picture/'
    font = 'png'
    view = v.TVisualizationModel(model,train_data_x,font,path1)
    view.save()