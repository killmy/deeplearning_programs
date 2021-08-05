import sys

# 添加程序路径(在win10下的开放)
# sys.path.append('E:/deeplearning_programs/')
# 添加程序路径(在linux下的路径)
sys.path.append('/media/g/学习资料盘/deeplearning_programs/')
import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.optim import optimizer
import data as d
import Model
import Visualization.visualization as v
import codelibrary.torchcode as c
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":
    # equipment
    device = c.gpuequipmet()
    print(device)
    #train_data 
    train_data = d.TrainData(d.x_train,d.y_train)
    batch_size = len(train_data)
    train_loader = DataLoader(dataset=train_data,
                                                batch_size=batch_size,
                                                shuffle=True)
    #params
    epochs = 60
    lr = 0.001
    # 是否加载提前训练的模型
    premodel = 1
    # model
    
    if premodel == 0:
        model = Model.LinearModel().to(device)
    elif premodel == 1:
        if os.path.exists('./linear/params.ckpt'):
            model = Model.LinearModel().to(device)
            model.load_state_dict(torch.load('./linear/params.ckpt'))
        elif os.path.exists('./linear/model.ckpt'):
            model = torch.load('./linear/model.ckpt').to(device)
    model = nn.Linear(1,1).to(device=device)
    model = Model.Linear_Model().to(device)
    loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    total_step = len(train_data)*epochs
    # len(train_loader)一共有多少个batch
    for epoch in range(epochs):
        for i,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            #forward 
            outputs = model(images)
            loss = criterion(outputs,labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_item_num = (epoch+1)*len(train_data)+i
            if (i+1) % len(train_loader)==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1,epochs,global_item_num,total_step,loss.item()))
    model.eval()
    for images,labels in train_loader:
        model = model.cpu()
        pred = model(images)
        images = images.numpy()
        labels = labels.numpy()
        pred = pred.detach().numpy()
        plt.plot(images,labels,'r--o',label='traget')
        plt.plot(images,pred,'b--o',label="pred")
        plt.legend()
        plt.show()
# save model
torch.save(model,'./linear/model.ckpt')
torch.save(model.state_dict(),'./linear/params.ckpt')

