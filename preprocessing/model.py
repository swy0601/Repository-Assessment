from torch import nn
import torchvision
from preprocessing.ascii.model import ascii
from preprocessing.codebert.model import codeBert
import torch




class Embedding(nn.Module):
    def __init__(self,num_classes=3,pretrained=False):
        super().__init__()
        self.backbone=torchvision.models.resnet18(pretrained=pretrained,num_classes=num_classes)
        self.backbone.fc=nn.Sequential()
        self.linear=nn.Sequential(nn.Linear(512+768,num_classes),nn.Softmax())

    def _forward_semantic(self,x:str):
        return ascii(x),codeBert(x)[1]

    def _forward_feature(self,x_ascii,x_codebert):
        return torch.concat((self.backbone(x_ascii.repeat((1,3,1,1))),x_codebert),dim=1)

    def _forward_classification(self,x):
        return self.linear(x)

    def forward(self,x_ascii,x_codebert):
        output=self._forward_feature(x_ascii,x_codebert)
        return self._forward_classification(output)
