
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset


# %%

java_data = pickle.load(open('java_data.pkl', 'rb'))

# %%

TEST_FOLDER = 0
BATCH_SIZE = 64
BALANCE = True
PRETRAINED = False
LR = 0.0001
EPOCH = 50

# %%

from preprocessing.model import Embedding
import torch.nn.functional as F


model = Embedding(num_classes=3, pretrained=PRETRAINED)
if torch.cuda.is_available():
    print('cuda available')
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



def padding(size,input):
    nn.functional.pad(input,(0,size[1]-input.shape[1]))



from dataloader import loadDataSet


train_dataset = None
test_dataset = None

if TEST_FOLDER == 1:
    train_dataset = loadDataSet(java_data, [1, 2], balance=BALANCE)
    test_dataset = loadDataSet(java_data, [0], balance=BALANCE)
elif TEST_FOLDER == 2:
    train_dataset = loadDataSet(java_data, [0, 2], balance=BALANCE)
    test_dataset = loadDataSet(java_data, [1], balance=BALANCE)
elif TEST_FOLDER == 3:
    train_dataset = loadDataSet(java_data, [0, 1], balance=BALANCE)
    test_dataset = loadDataSet(java_data, [2], balance=BALANCE)
else:
    train_dataset = loadDataSet(java_data, [0, 1,2], balance=BALANCE)


def collate(batch):
    x1 = torch.stack([i[0][0] for i in batch])
    x2 = torch.stack([i[0][1] for i in batch])
    y=torch.tensor([i[1] for i in batch])
    return (x1,x2),y

train_loader = DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2,
    drop_last=True,
    collate_fn=collate
)


test_loader=None
if test_dataset!=None:
    test_loader = DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        # num_workers=2,
        collate_fn=collate
    )


# %%


# %%

# plt.figure(figsize=(10, 10))


def evaluate(data_loader):
    if data_loader==None:
        return
    avg_acc = 0
    count = 0
    model.eval()
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x0 = batch_x[0].float()
        batch_x1 = batch_x[1].float()
        batch_y = batch_y.long()
        if torch.cuda.is_available():
            batch_x0=batch_x0.cuda()
            batch_x1=batch_x1.cuda()
            batch_y = batch_y.cuda()
        output = model(batch_x0,batch_x1)
        max_index = torch.argmax(output, dim=1)
        accuracy = torch.sum(max_index == batch_y).item() / len(batch_y)
        avg_acc += accuracy
        count += 1
    return avg_acc / count


def train(data_loader):
    global loss
    model.train()
    avg_loss = 0
    count = 0
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x0 = batch_x[0].float()
        batch_x1 = batch_x[1].float()
        batch_y = batch_y.long()
        if torch.cuda.is_available():
            batch_x0=batch_x0.cuda()
            batch_x1=batch_x1.cuda()
            batch_y = batch_y.cuda()
        y_pred= model(batch_x0,batch_x1)
        l=nn.functional.cross_entropy(y_pred,batch_y)
        # l = loss(y_proj, y_pred, batch_y)
        # l = l[0] + l[1]
        # loss = nn.CrossEntropyLoss()(output, batch_y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        avg_loss += l.item()
        count += 1
    return avg_loss / count


train_accuracy = evaluate(train_loader)
test_accuracy = evaluate(test_loader)
print('TRAIN ACCURACY: {}; TEST ACCURACY: {}'.format(train_accuracy, test_accuracy))

for epoch in range(EPOCH):
    l = train(train_loader)
    train_accuracy = evaluate(train_loader)
    test_accuracy = evaluate(test_loader)
    print('EPOCH: {}; LOSS: {}; TRAIN ACCURACY: {}; TEST ACCURACY: {}'.format(epoch, l, train_accuracy, test_accuracy))

torch.save(model.state_dict(), "embedding")