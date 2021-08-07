# codeing:utf-8
# add path
import sys
sys.path.append("/media/g/学习资料盘/deeplearning_programs/")
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import data as T
import Visualization.visualization as v
import Model
import codelibrary.torchcode as ct
import torch.optim  as o
# devince
devince = ct.gpuequipmet()


# params and others
class Params():
    data_path = '/media/g/学习资料盘/dataset/MNIST/data'
    log_path = './logistic/log'
    model_path = './logistic/model.ckpt'
    model_params_path = './logistic/params.ckpt'
    input_size = 28*28 #784
    num_classes = 10
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
params = Params()

# logger
logger = v.TVisualizationTrain(params.log_path)

# data
train_data = T.MnistTrainData(params.data_path)
train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=params.batch_size, 
                                           shuffle=True)
test_data = T.MnistTestData(params.data_path)
test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                          batch_size=params.batch_size, 
                                          shuffle=False)
# model 
model = Model.LogisticModel(params.input_size,params.num_classes).to(devince)
load_model = ct.LoadModel()
model = load_model.add_params(model,params.model_params_path)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = o.SGD(model.parameters(),lr=params.learning_rate)

# train
total_step = len(train_loader)*params.num_epochs
for epoch in range(params.num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.reshape(-1,params.input_size)
        images = images.to(devince)
        labels = labels.to(devince)

        # forward
        out = model(images)
        loss = criterion(out,labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #show 
        global_iter_num  = epoch*len(train_loader)+i+1
        # 一整个数据打印一次
        if (i+1)%len(train_loader)==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
            .format(epoch+1, params.num_epochs, global_iter_num, total_step, loss.item()))
            logger.add_scalar(global_iter_num,train_loss=loss.item())
            # 照片太多容易出问题
            logger.add_image("images",images,12,global_iter_num)
            logger.add_histogram('params',model,global_iter_num)

# test
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.reshape(-1,params.input_size)
        images = images.to(devince)
        labels = labels.to(devince)
        outputs = model(images)
        _,predicted = torch.max(outputs,1)
        total +=labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), params.model_params_path)

