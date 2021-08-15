import sys

from torch.nn.modules import loss
sys.path.append("/media/g/学习资料盘/deeplearning_programs/")
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data
import Visualization.visualization as v
import codelibrary.torchcode as ct
from torch.utils.data import DataLoader, dataset
import Model
from torch import optim
if __name__ == "__main__":
    # device
    device = ct.gpuequipmet()

    # logger
    logger = v.TVisualizationTrain('./cnn/simple_model/log/')
    # params
    class Params():
        num_epochs = 5
        num_classes = 10
        batch_size = 100
        learning_rate = 0.001
    params = Params()
    #data
    train_data = data.train_dataset
    train_loader = DataLoader(dataset=train_data,
                                                                    batch_size=params.batch_size,
                                                                    shuffle=True)
    test_data = data.test_dataset
    test_loader = DataLoader(dataset=test_data,
                                                                                batch_size=params.batch_size,
                                                                                shuffle=False)

    # model 
    model = Model.SimpleCNNModel(10).to(device=device)

    # loss and optimizer
    critertion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=params.learning_rate)

    # train
    global_num_item = params.num_epochs*len(train_loader)
    for epoch in range(params.num_epochs):
        for i,(images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = critertion(outputs,labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #now num
            now_item = epoch*len(train_loader)+(i+1)

            if (i+1)%100 ==0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1,params.num_epochs,now_item,global_num_item,loss.item()))
                # img = torchvision.utils.make_grid(images,nrow=12)
                # print(img.size())
                show_images = images.cpu()
                show_labels = labels.cpu()
                logger.add_scalar(now_item,train_loss = loss.item())
                logger.add_image('train_image',show_images,12,now_item)
                logger.add_histogram("params",model,now_item)
                correct,total = ct.ClassAcc().predict(model,test_loader,device)
                print('accuracy ia %{}'.format(100 * correct / total))
                logger.add_scalar(now_item,accuracy=(correct/total))
                