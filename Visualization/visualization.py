import hiddenlayer as h
from torch.utils import data
from torchviz import make_dot
from tensorboardX import SummaryWriter
# show model
class HVisualizationModel():
    'simply show model'
    def __init__(self,model,data,path,theme) -> None:
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
    def __init__(self) -> None:
        pass

