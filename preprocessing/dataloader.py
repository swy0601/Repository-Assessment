import torch
from torch.utils.data import Dataset
from torch.nn import functional as f


class CodeDataset(Dataset):

    def __init__(self,data_ascii,data_codebert,label):
        self.x1 = data_ascii
        self.x2=data_codebert
        self.y = label
        self.len = len(self.x1)

    def __getitem__(self, index):
        return (self.x1[index],self.x2[index] ),self.y[index]

    def __len__(self):
        return self.len


from ascii.model import ascii
from codebert.model import codeBert

def handle(lst,method=ascii):
    results=[]
    for i in lst:
        results.append(method(i))
    return results


def loadDataSet(java_data, folder=[0, 1], balance=False,size=(128,128)):
    data_ascii = []
    data_codebert=[]
    label = []
    # model=torch.nn.AdaptiveAvgPool2d(size)
    # model=model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    def model(x,size=size):
        x=x.squeeze(0)
        x=x[:size[0],:size[1]]
        if x.shape[1]<size[1]:
            x=f.pad(x,(0,size[1]-x.shape[1]))
        if x.shape[0]<size[0]:
            em=torch.zeros(size=(size[0]-x.shape[0],size[1]))
            x=torch.concat((x,em),0)
        return x.unsqueeze(0)

    _method1=ascii
    _method2=codeBert

    def method1(x):
        return model(_method1(x).unsqueeze(0))

    def method2(x):
        return _method2(x).squeeze(0)

    for i in folder:
        data_ascii += handle(java_data[i]['Neutral'],method1)
        data_codebert += handle(java_data[i]['Neutral'], method2)
        label += [0] * java_data[i]['Neutral'].__len__()

        data_ascii += handle(java_data[i]['Readable'],method1)
        label += [1] * java_data[i]['Readable'].__len__()
        data_codebert += handle(java_data[i]['Readable'], method2)

        data_ascii += handle(java_data[i]['Unreadable'],method1)
        label += [2] * java_data[i]['Unreadable'].__len__()
        data_codebert += handle(java_data[i]['Unreadable'], method2)
        if balance:
            data_ascii += handle(java_data[i]['Readable'],method1)
            label += [1] * java_data[i]['Readable'].__len__()
            data_codebert += handle(java_data[i]['Readable'], method2)
            data_ascii += handle(java_data[i]['Unreadable'],method1)
            label += [2] * java_data[i]['Unreadable'].__len__()
            data_codebert += handle(java_data[i]['Unreadable'], method2)
    return CodeDataset(data_ascii, data_codebert,label)