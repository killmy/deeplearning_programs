import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import torchvision
# data
# 一般label名字和图片的顺序对应，如json中序号0代白文件中编号为0的picture或者
# 其他，对应的lable就是字典'0'对应的label
# "0": ["n01440764", "tench"] picture 0的label就是'tench
# data path
data_path = './adversarial/FGMS/easy/data/imagenet_class_index.json'
class_idx = 0
# 读取json文件,获得class and class_idex
if os.path.exists(data_path):
    # read json文件
    class_idx = json.load(open(data_path))
else:
    pass
if class_idx!=0:
    # read in order
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]    
else:
    print('no data')
    pass
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
])
# readdata
def image_folder_custom_label(root, transform, custom_label) :
    
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']
    
    old_data = dsets.ImageFolder(root = root, transform = transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root = root, transform = transform, 
                                 target_transform = lambda x : custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx
    return new_data

normal_data = image_folder_custom_label(root = './adversarial/FGMS/easy/data/', transform = transform, custom_label = idx2label)
normal_loader = Data.DataLoader(normal_data, batch_size=1, shuffle=False)

def imshow(img, title):
        npimg = img.numpy()
        # fig = plt.figure(figsize = (5, 15))
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.title(title)
        plt.show()
        
if __name__ == "__main__":
    print(normal_data[0])
    def imshow(img, title):
        npimg = img.numpy()
        # fig = plt.figure(figsize = (5, 15))
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.title(title)
        plt.show()
    # image,label = normal_data[0]
    # print(image.shape)
    # for images,labels in normal_loader:
    #     print(images.shape)

    # for images,labels in normal_loader:
    #     print(class_idx[str(labels.numpy()[0])][1])

    normal_iter = iter(normal_loader)
    images, labels = normal_iter.next()

    print("True Image & True Label")
    imshow(torchvision.utils.make_grid(images, normalize=True), [normal_data.classes[i] for i in labels])