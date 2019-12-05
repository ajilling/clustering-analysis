'''
    Author: Adam Jilling
    Build neural net and test/train using LOOCV
    Output results to .csv file
'''

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
from torch import optim

tensors = pd.read_csv("../Results/tensors.csv", header=None, usecols=[*range(0, 34)])
D_in = 32
D_out = 2
lr = float(sys.argv[1]) # command line argument
wd = float(sys.argv[2]) # command line argument
n_iter = int(sys.argv[3]) # command line argument
max_dataset = int(sys.argv[4]) # command line argument
scaler_x = StandardScaler()
scaler_y = StandardScaler()
loss_fn = torch.nn.MSELoss()
total_train_loss = 0.0
total_test_loss = 0.0
predictions = pd.DataFrame(columns=['pred runtime', 'pred performance'])

def create_model():
    model = torch.nn.Sequential(
        nn.Linear(D_in, 24),
        nn.ReLU(),
        nn.Linear(24, 8),
        nn.ReLU(),
        nn.Linear(8, D_out),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return model, optimizer

def split_data(d_idx):
    l = d_idx * 7
    h = (d_idx+1) * 7
    test_x = tensors.iloc[l:h, :32].values
    test_y = tensors.iloc[l:h, 32:].values
    temp = tensors.drop(tensors.index[l:h])
    train_x = temp.iloc[:, :32].values
    train_y = temp.iloc[:, 32:].values
    scaler_x.fit(train_x)
    scaler_y.fit(train_y)
    return torch.FloatTensor(scaler_x.transform(train_x)), torch.FloatTensor(scaler_y.transform(train_y)), torch.FloatTensor(scaler_x.transform(test_x)), torch.FloatTensor(scaler_y.transform(test_y))

def train_test():
    results = []
    # LOOCV
    for i in range(max_dataset):
        model, optimizer = create_model()
        model.train() # put in training mode
        tr_x, tr_y, te_x, te_y = split_data(i)
        # train model
        for t in range(n_iter):
            optimizer.zero_grad()
            tr_pred = model(tr_x)
            loss = loss_fn(tr_pred, tr_y)
            if not t % 200:
                print('Dataset: {}, Iter: {}, Loss: {}'.format(i, t, loss.item()))
            loss.backward()
            optimizer.step()
        # test model
        model.eval() # put in testing mode
        te_pred = model(te_x)
        train_loss = loss.item()
        test_loss = loss_fn(te_pred,te_y).item()
        print('For dataset {}, loss is {}'.format(i, test_loss))
        results.append((te_pred, te_y, train_loss, test_loss))
    return results

if __name__ == '__main__':
    results = train_test()
    np.set_printoptions(suppress=True)
    for pred, act, tr_loss, te_loss in results:
        print('Pred:')
        pp = pred.detach().numpy()
        predictions = predictions.append(pd.DataFrame(pp), ignore_index=True)
        print(pp)
        print(np.argmin(pp[:,0]), np.argmax(pp[:,1]))
        print('Act:')
        aa = act.numpy()
        print(aa)
        print(np.argmin(aa[:,0]), np.argmax(aa[:,1]))
        print('Train Loss:')
        print(tr_loss)
        total_train_loss += tr_loss
        print('Test Loss:')
        print(te_loss)
        total_test_loss += te_loss
    print('---------------')
    print('Learning Rate: {}'.format(lr))
    print('Weight Decay: {}'.format(wd))
    print('Average Train Loss: {}'.format(total_train_loss/max_dataset))
    print('Average Test Loss: {}'.format(total_test_loss/max_dataset))

    pd.DataFrame(predictions).to_csv("~/Desktop/predictions.csv", index = False, header = False)
