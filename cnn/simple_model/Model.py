import torch
import torch.nn as nn
import data
class SimpleCNNModel(nn.Module):
    def __init__(self,num_classes=10):
        super(SimpleCNNModel,self).__init__()
        self.layer1 = nn.Sequential(
            # 一般卷积保持原图大小不变--输入28*28
            #(image.size+2*padding-kernel_size)+stride
            nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),
            # batch normal 防止某些数据影响过大(用relu方式最好做一下batch normal)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 最大池化方法
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc = nn.Linear(7*7*32,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    import sys
    sys.path.append("/media/g/学习资料盘/deeplearning_programs/")
    import Visualization.visualization as v
    import hiddenlayer as h
    model = SimpleCNNModel(10)
    input_data,label = data.train_dataset[0]
    input_data = torch.unsqueeze(input_data,dim=0)
    print(model)
    path = './cnn/simple_model/picture/model1'
    theme = "blue"
    view = v.HVisualizationModel(model=model,data=input_data,path=path,theme=theme)
    view.save()
    path1 = './cnn/simple_model/picture'
    font = 'png'
    view = v.TVisualizationModel(model,input_data,font,path1)
    view.save()

