import hiddenlayer as h
from hiddenlayer import history
from matplotlib.pyplot import step
from torch.nn.modules import loss
from torch.utils import data
from torchviz import make_dot
from tensorboardX import SummaryWriter
import torchvision
import time
# 目前只尝试了这几种可视化，后面有时间再增加，先用着
# show model
class HVisualizationModel():
    'simply show model'
    def __init__(self,model,data,path,theme) -> None:
        'model,data,path,theme'
        self.vis_graph = h.build_graph(model,data)
        self.vis_graph.theme = h.graph.THEMES[theme].copy()
        self.path = path
    def save(self):
        self.vis_graph.save(self.path)

class TVisualizationModel():
    'show model'
    def __init__(self,model,x_data,format,path) -> None:
        self.model = model
        self.x_data = x_data.requires_grad_(True) 
        self.y = self.model(self.x_data)
        self.MyConvNetVis = make_dot(self.y,params=
            dict(list(self.model.named_parameters()) + [('x', self.x_data)]))
        self.MyConvNetVis.format = format
        self.MyConvNetVis.directory = path
    def save(self)->None:
        self.MyConvNetVis.view()

# show train
class TVisualizationTrain():
    'imcomplete code'
    def __init__(self,path) -> None:
        self.logger = SummaryWriter(log_dir=path)
    def add_scalar(self,step,**kwargs)->None:
        '添加常数'
        for key,value in kwargs.items():
            self.logger.add_scalar(key,value,global_step=step)
    # def add_scalar(self,name,param,step)->None:
    #     '添加常数'
    #     self.logger.add_scalar(name,param,global_step=step)
    def add_image(self,name,images,image_size,step):
        '添加当前batch的训练图片'
        image = torchvision.utils.make_grid(images,nrow=image_size)
        self.logger.add_image(name,image,global_step=step)
    def add_histogram(self,name,model,step):
        '参数直方图'
        model = model.cpu()
        for name ,param in model.named_parameters():
                #https://blog.csdn.net/moshiyaofei/article/details/90519430
                #要先转化为cpu才能转化为numpy
                self.logger.add_histogram(name, param.data.numpy(), global_step=step)
        model = model.cuda()

# class HVisualizationTrain():
#     def __init__(self) -> None:
#         self.history = h.History()
#         self.canvas = h.Canvas()
#     def canves_scalar(self,step,**kwargs):
#         self.history.log(step,**kwargs)
#         for key in kwargs:
#             self.canvas.draw_plot(self.history[key])
#     def canves_weight(self,step,**kwargs):
#         self.history.log(step,**kwargs)
#         for key in kwargs:
#             self.canvas.draw_hist(self.history[key])
