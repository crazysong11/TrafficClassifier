import random
import time
import net

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import torch.utils.data.dataloader as Data
from torch.utils.tensorboard import SummaryWriter


def MaxMinNormalization(num, Max, Min):
    num = (num - Min) / (Max - Min)
    return num


def Z_ScoreNormalization(num, mu, sigma):
    num = (num - mu) / sigma
    return num


def getThreshold(tes):
    kk = []
    count = 0
    for i in range(len(tes)):
        count += tes[i][0]
        kk.append(tes[i][0])
    d = 0.7
    return min(kk) + (max(kk) - min(kk)) * d


def test(model, Xtest_tensor, Ytest_tensor):
    Xtest_tensor = Xtest_tensor.numpy()
    for k in range(len(Xtest_tensor)):
        Xtest_tensor[k] = [Z_ScoreNormalization(_, Xtest_tensor[k].mean(), Xtest_tensor[k].std()) for _ in
                           Xtest_tensor[k]]
    Xtest_tensor = torch.tensor(torch.from_numpy(Xtest_tensor), dtype=torch.float32)
    X_data = Xtest_tensor.reshape(-1, 1, 36)
    Ypred = model(X_data)
    threshold = getThreshold(Ypred.detach().numpy())
    Ypred_binary = torch.where(Ypred > threshold, torch.tensor([1.0]), torch.tensor([0.0]))
    correct = (Ypred_binary == Ytest_tensor.reshape(-1, 1)).sum()
    total = len(Ytest_tensor)
    accuracy = correct.double().item() / total

    TP = 0
    TN = 0
    FN = 0
    FP = 0
    with torch.no_grad():
        # TP    predict 和 label 同时为1
        TP += ((Ypred_binary.data == 1) & (Ytest_tensor.data == 1)).cpu().sum()
        # TN    predict 和 label 同时为0
        TN += ((Ypred_binary.data == 0) & (Ytest_tensor.data == 0)).cpu().sum()
        # FN    predict 0 label 1
        FN += ((Ypred_binary.data == 0) & (Ytest_tensor.data == 1)).cpu().sum()
        # FP    predict 1 label 0
        FP += ((Ypred_binary.data == 1) & (Ytest_tensor.data == 0)).cpu().sum()

    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    return accuracy, p, r, F1


def get_model():
    model = net.Model()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    return model, opt


seed = int(time.time_ns())
random.seed(seed)

np.set_printoptions(suppress=True, precision=20, threshold=10, linewidth=40)  # np禁止科学计数法显示
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # pd禁止科学计数法显示

print("--------------read data---------------")
df_all = pd.read_csv(r'result_real.csv')

df_all_labels = df_all["Label"].copy()
df_all.drop(["Label"], axis=1, inplace=True)
print("--------------read data done---------------")


print("--------------separate data---------------")
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_all, df_all_labels, test_size=0.3, random_state=0)
print("--------------separate data done---------------")


Xtrain_tensor = torch.as_tensor(torch.from_numpy(Xtrain.values), dtype=torch.float32)
Ytrain_tensor = torch.tensor(Ytrain.values.astype(int), dtype=torch.float32)
Xtest_tensor = torch.as_tensor(torch.from_numpy(Xtest.values), dtype=torch.float32)
Ytest_tensor = torch.tensor(Ytest.values.astype(int), dtype=torch.float32)

lr = 0.005

model, optim = get_model()

loss_func = nn.BCELoss()
batch = 256
no_of_batch = len(df_all) // batch
epoch = 100
best = 0
# 定义tensorboard可视化
writer = SummaryWriter('./boardlog')


print("--------------Train Start---------------")
for i in range(epoch):
    train_acc = 0
    for j in range(no_of_batch):
        start = j * batch
        end = start + batch
        x = Xtrain_tensor[start:end]
        y = Ytrain_tensor[start:end]
        x = x.numpy()
        for k in range(len(x)):
            x[k] = [Z_ScoreNormalization(_, x[k].mean(), x[k].std()) for _ in x[k]]
        x = torch.as_tensor(torch.from_numpy(x), dtype=torch.float32)
        X_data = x.reshape(-1, 1, 36)
        yPred = model(X_data)
        loss = loss_func(yPred, y.reshape(-1, 1))
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        print('epoch : ', i, ' loss: ', loss_func(model(Xtrain_tensor.reshape(-1, 1, 36)), Ytrain_tensor.reshape(-1, 1)).data.item())
    accuracy, precision, recall, F1 = test(model, Xtest_tensor, Ytest_tensor)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    print('Precision: {:.2f}%'.format(precision * 100))
    print('Recall: {:.2f}%'.format(recall * 100))
    print('F1: {:.2f}%'.format(F1 * 100))
    # 记录train loss和测试集上的识别准确度
    writer.add_scalar('train_loss', loss_func(model(Xtrain_tensor.reshape(-1, 1, 36)), Ytrain_tensor.reshape(-1, 1)).data.item(), global_step=i, walltime=None)
    writer.add_scalar('accuracy', accuracy, global_step=i, walltime=None)
    if accuracy > best:
        print('bestModel: epoch{}'.format(i))
        torch.save(model, 'net.pth')
        best = accuracy
        bestModel = i
    else:
        print('bestModel: epoch' + str(bestModel))
    print("-----------------------------")
print("--------------Train Done---------------")

