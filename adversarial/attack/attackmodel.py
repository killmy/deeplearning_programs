import torch
import torch.nn as nn

class Attack():
    def __init__(self,model,device) -> None:
        self.model = model
        self.device = device
    def fgsm_attacks_add(self,criterion,images,labels,eps):
        '增加真实label的不可能性'
        images = images.to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True
        outputs = self.model(images)
        self.model.zero_grad()
        loss = criterion(outputs, labels).to(self.device)
        loss.backward()
        attack_images = images + eps*torch.sign(images.grad)
        attack_images = torch.clamp(attack_images, 0, 1)
        return attack_images
