import numpy as np
import torch
import torch.nn.functional as F

def ascii(x: str):
    """
    :param x:
     e.g.
         line1 ....   \n
         line2 ....   \n
         line3 ....   \n
    :return:
    """

    def Ascii(content):
        result = ''
        prev = ''
        for c in content:
            if c == '\n':
                if prev == '\n':
                    continue
                else:
                    prev = c
                    result += '\n,'
            else:
                result += str(ord(c)) + ','
                prev = c
        return result

    def clean_file(file_data):
        # file_data = file_data.replace("\n", " ")
        file_data = file_data.replace("\t", " ")
        file_data = file_data.replace("\r", " ")
        file_data = file_data.replace("\f", " ")
        file_data = file_data.replace("  ", " ")
        return file_data

    data = clean_file(x)
    data = [i.split(",") for i in Ascii(data).split("\n") if i != ""]
    data = [[int(term) for term in line if term != ""] for line in data if len(line) > 1]
    max_len = max([len(line) for line in data])
    data = [line + [-1] * (max_len - len(line)) for line in data]
    data = np.array(data)
    # normalize
    data = (data - data.min()) / (data.max() - data.min())
    return torch.tensor(data)



def temp_ascii(x):
    """
    :param x:
     e.g.
         line1 ....   \n
         line2 ....   \n
         line3 ....   \n
    :return:
    """

    def Ascii(content):
        result = ''
        prev = ''
        for c in content:
            if c == '\n':
                if prev == '\n':
                    continue
                else:
                    prev = c
                    result += '\n,'
            else:
                result += str(ord(c)) + ','
                prev = c
        return result

    def clean_file(file_data):
        # file_data = file_data.replace("\n", " ")
        file_data = file_data.replace("\t", " ")
        file_data = file_data.replace("\r", " ")
        file_data = file_data.replace("\f", " ")
        file_data = file_data.replace("  ", " ")
        return file_data

    data = clean_file(x)
    data = [i.split(",") for i in Ascii(data).split("\n") if i != ""]
    data = [[int(term) for term in line if term != ""] for line in data if len(line) > 1]
    max_len = max([len(line) for line in data])
    data = [line + [-1] * (max_len - len(line)) for line in data]
    def padding(x,size=(256,256)):
        x=x.squeeze(0)
        x=x[:size[0],:size[1]]
        if x.shape[1]<size[1]:
            x=F.pad(x,(0,size[1]-x.shape[1]))
        if x.shape[0]<size[0]:
            em=torch.zeros(size=(size[0]-x.shape[0],size[1]))-1
            x=torch.concat((x,em),0)
        return x.unsqueeze(0)
    data = torch.tensor(data)
    data=padding(data,size=(50,305))

    # normalize
    data = data/127
    return data.numpy()