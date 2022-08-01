import copy
import os.path
import random

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
import kmeans
from sklearn.metrics import silhouette_score
from model.gtn.model import GTN
from model.model import Model
from utils import load_data, EarlyStopping, load_test_data
DATA_PATH="D:\Project\Graph\mlkit.pkl"

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1


def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    # g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    # val_mask, test_mask = load_data(args['dataset'])

    # if hasattr(torch, 'BoolTensor'):
    #     train_mask = train_mask.bool()
    #     val_mask = val_mask.bool()
    #     test_mask = test_mask.bool()
    #
    # features = features.to(args['device'])
    # labels = labels.to(args['device'])
    # train_mask = train_mask.to(args['device'])
    # val_mask = val_mask.to(args['device'])
    # test_mask = test_mask.to(args['device'])

    batch_size = 100
    batch_total = 10000

    g, features, positive_samples, negative_samples,files= load_test_data(DATA_PATH)

    cluster, center = kmeans.kmeans(features, 3)
    score = silhouette_score(features.cpu().detach(), cluster.cpu().detach())
    print("score:",score)

    from model.model import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model = Model(meta_paths=[['fa', 'af']],
                  embedding_size=1280,
                  hidden_size=args['hidden_units'],
                  out_size=128,
                  num_heads=args['num_heads'],
                  dropout=args['dropout']).to(device)
    # g = [graph.to(args['device']) for graph in g]
    # g=g[0]
    model = model.to(device)

    best_score=0
    best_model=None

    g = g.to(device)
    features = features.to(device)
    # if args['hetero']:
    #     from model_hetero import HAN
    #     model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
    #                 in_size=features.shape[1],
    #                 hidden_size=args['hidden_units'],
    #                 out_size=num_classes,
    #                 num_heads=args['num_heads'],
    #                 dropout=args['dropout']).to(args['device'])
    #     g = g.to(args['device'])
    # else:
    #     from model import HAN
    #     model = HAN(num_meta_paths=len(g),
    #                 in_size=features.shape[1],
    #                 hidden_size=args['hidden_units'],
    #                 out_size=num_classes,
    #                 num_heads=args['num_heads'],
    #                 dropout=args['dropout']).to(args['device'])
    #     g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(str(os.path.basename(DATA_PATH)).split('.')[0],patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                 weight_decay=args['weight_decay'])

    import random

    test_samples=random.sample(range(features.shape[0]),int(features.shape[0]/3))
    _test_samples=set(test_samples)
    _positive_samples=[]
    _negative_samples=[]

    train_samples=list(set(range(features.shape[0]))-_test_samples)


    for i in positive_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _positive_samples.append(i)

    for i in negative_samples:
        if i[0] not in _test_samples and i[1] not in _test_samples:
            _negative_samples.append(i)


    print(positive_samples.__len__(), _positive_samples.__len__())
    print(negative_samples.__len__(), _negative_samples.__len__())

    positive_samples=_positive_samples
    negative_samples=_negative_samples

    valid_samples=test_samples[0:int(test_samples.__len__()/2)]
    test_samples=test_samples[int(test_samples.__len__()/2):]


    batch_total=int(min(positive_samples.__len__(),negative_samples.__len__(),batch_total)/batch_size)*batch_size
    best_epoch=0

    for epoch in range(50):
        model.train()
        random.shuffle(positive_samples)
        random.shuffle(negative_samples)


        total_loss=0
        for i in tqdm(range(int(batch_total/batch_size))):
            output = model(g, features)
            p = positive_samples[i * batch_size:(i + 1) * batch_size]
            n = negative_samples[i * batch_size:(i + 1) * batch_size]
            a, b, c,d = [], [], [],[]
            for i in p:
                a.append( i[0])
                b .append( i[1])
            for i in n:
                c.append( i[0])
                d.append(i[1])
            l1=model._forward_train_positive(output,a,b)
            l2=model._forward_train_negative(output,c,d)
            loss=l1+l2
            # loss.requires_grad_(True)
            optimizer.zero_grad()
            total_loss+=loss.item()
            loss.backward()
            optimizer.step()


        # train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        # val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)
        # early_stop = stopper.step(val_loss.data.item(), val_acc, model)
        model.eval()
        output = model(g, features)

        _output=torch.tensor([output[i].cpu().detach().numpy() for i in train_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, 3, device=device)
        score1 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())


        _output=torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
        cluster, center = kmeans.kmeans(_output, 3, device=device)
        score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

        print("EPOCH:{}  score train:{} score valid:{}  average_loss:".format(epoch, score1,score2), total_loss / (batch_total / batch_size))
        # print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
        #       'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
        #     epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))
        if score2>best_score and epoch>39:
            best_model=copy.deepcopy(model)
            best_score=score2
            best_epoch=epoch
        stopper.save_checkpoint(model)
        # if early_stop:
        #     break

    best_model.eval()
    output = best_model(g, features)

    _output = output
    cluster, center = kmeans.kmeans(_output, 3, device=device)
    score0 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())



    _output = torch.tensor([output[i].cpu().detach().numpy() for i in valid_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, 3, device=device)
    score2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, 2, device=device)
    test2 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    _cluster, center = kmeans.kmeans(_output, 3, device=device)
    test3 = silhouette_score(_output.cpu().detach(), _cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, 4, device=device)
    test4 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, 5, device=device)
    test5 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    _output = torch.tensor([output[i].cpu().detach().numpy() for i in test_samples]).to(device)
    cluster, center = kmeans.kmeans(_output, 6, device=device)
    test6 = silhouette_score(_output.cpu().detach(), cluster.cpu().detach())

    lst1 = [[], [], []]
    lst2 = [[],[],[]]
    for i in range(cluster.__len__()):
        lst1[_cluster[i]].append(files[test_samples[i]])

    for i in range(cluster.__len__()):
        lst2[_cluster[i]].append(test_samples[i])

    m=g.adjacency_matrix(etype="fa")

    d1,d2,d3={},{},{}

    for i in lst2[0]:
        for ii in m[i].coalesce().indices().tolist()[0]:
            d1[ii]=d1.get(ii,0)+1

    for i in lst2[1]:
        for ii in m[i].coalesce().indices().tolist()[0]:
            d2[ii]=d2.get(ii,0)+1

    for i in lst2[2]:
        for ii in m[i].coalesce().indices().tolist()[0]:
            d3[ii]=d3.get(ii,0)+1


    _d1,_d2,_d3={},{},{}

    for i in d1.keys():
        _d1.setdefault(d1[i],[])
        _d1[d1[i]].append(i)

    for i in d2.keys():
        _d2.setdefault(d2[i],[])
        _d2[d2[i]].append(i)

    for i in d3.keys():
        _d3.setdefault(d3[i],[])
        _d3[d3[i]].append(i)

    # print("best_epoch:{}   score total:{}  score test:{}   score valid:{} ".format(best_epoch,score0,score3,score2))
    print("best_epoch:{}   score total:{} score valid:{} score test2:{} score test3:{}  score test4:{} score test5:{} score test6:{}".format(best_epoch, score0, score2, test2,test3,test4,test5,test6))
    torch.save(best_model,DATA_PATH.split("\\")[-1][:-4]+"_model")
    stopper.load_checkpoint(model)
    # test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn)
    # print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
    #     test_loss.item(), test_micro_f1, test_macro_f1))




if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-d','--data',type=str,default='.',help='data path')
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)
    # DATA_PATH=args.__getitem__('data')
    main(args)
