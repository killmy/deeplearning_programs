import os
import torch
import torch.nn as nn
import torch.optim as optim
import data as d
import sys
sys.path.append('/media/g/学习资料盘/deeplearning_programs/')
import codelibrary.torchcode as ct
from torchvision import models
import adversarial.attack.attackmodel as a
import matplotlib.pyplot as plt
import torchvision
# devince
devince = ct.gpuequipmet()
# params
eps = 0.007
# 载入有预训练的模型
model = models.inception_v3(pretrained=True).to(devince)
print("True Image & Predicted Label")
# model.eval()
# correct = 0
# total = 0
# for images,labels in d.normal_loader:
#     images = images.to(devince)
#     labels = labels.to(devince)
#     outputs = model(images)
#     _, pre = torch.max(outputs.data, 1)
#     total +=labels.size(0)
#     correct += (pre == labels).sum().item()
correct,total = ct.ClassAcc().predict(model,d.normal_loader,devince)
print('Accuracy of the network : {} %'.format(100 * correct / total))
# attack
# loss
criterion = nn.CrossEntropyLoss()
model.eval()
for images, labels in d.normal_loader:
    # images = images.to(devince)
    # print(images.requires_grad)
    # images.requires_grad = True
    # print(images.requires_grad)
    # output = model(images)
    # model.zero_grad()
    # loss  = criterion(output,labels).to(devince)
    # loss.backward()
    
    images = a.Attack(model,devince).fgsm_attacks_add(criterion,images,labels,eps)
    labels = labels.to(devince)
    correct,total= ct.ClassAccTwo().predict(model,images,labels,devince)
    print('Accuracy of the network : {} %'.format(100 * correct / total))
    predicted = ct.classPredict().predict(model,images,devince)
    print(predicted)
    d.imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), [d.normal_data.classes[i] for i in predicted])
    
