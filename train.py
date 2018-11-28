import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import sys
import pickle
import copy
import argparse
import itertools
import make_data as make_data
from model import *
from tensorboardX import SummaryWriter
from random import shuffle
import time
from sklearn.model_selection import KFold, StratifiedKFold

out_size = 3
max_epoch = 500
dropout = 0#0.5 #0.1
lr = 0.01 #0.01
weight_decay = 0# 0.001 # L2 penalty
best_model = True
cuda_flag = False
# -----------------------------------
loss_function = nn.CrossEntropyLoss() ###
# -----------------------------------

def variable(l):
    if cuda_flag == True:
        return Variable(torch.from_numpy(np.array(l))).cuda()
    else:
        return Variable(torch.from_numpy(np.array(l)))

# -----------------------------------
if __name__ == "__main__":
    # File Path
    # -load-
    DATA           = 'data/np_data.pkl'

    # Early Stop
    history_valid_loss = [10]
    earlystop_counter = 0
    earlystop_threshold = 10

    best_train_info = [0]*2 #acc, loss
    best_dev_info = [0]*2 #acc, loss

    # Prepare training & testing data 
    n, p = pickle.load(open(DATA, 'rb'))
    n_train_size = int(0.8 * (len(n)))
    n_test_size = len(n) - n_train_size
    p_train_size = int(0.8 * (len(p)))
    p_test_size = len(p) - p_train_size

    n_train_data, n_test_data = Data.random_split(n, [n_train_size, n_test_size])
    p_train_data, p_test_data = Data.random_split(p, [p_train_size, p_test_size])

    train_data = list(itertools.chain(n_train_data, p_train_data))
    test_data = list(itertools.chain(n_test_data, p_test_data))
    shuffle(train_data)
    shuffle(test_data)
    
    print("preparing training dataset...")
    signal, target = [], []
    for data in train_data:
        signal.append(data[0])
        target.append(data[1])
     
    # testing hyperparemeter
    minibatchs = [1, 2, 4, 8, 16, 32]

    print("start training...")
    # ------training------
    np_signal = np.asarray(signal)
    np_target = np.asarray(target)
    skf = StratifiedKFold(n_splits=10)
    history_valid_loss = []
    history_valid_acc = []
    for minibatch in minibatchs:
        fold_idx = 0
        valid_average_loss, valid_average_acc = [0.0]*10, [0]*10
        for train_idx, val_idx in skf.split(np_signal, np_target):
            #print(train_idx, val_idx)
            x_train, x_val = np_signal[train_idx], np_signal[val_idx]
            y_train, y_val = np_target[train_idx], np_target[val_idx]
            train_dataset = make_data.myDataset(signal=x_train, target=y_train)
            trainloader =  Data.DataLoader(dataset=train_dataset, batch_size=minibatch, shuffle=True, collate_fn=make_data.my_collate, num_workers=0)
            print("length of trainloader:", len(trainloader))    
            start_time = time.time()
            count, pcount = [0]*out_size, [0]*out_size
            total_loss, total_acc = 0.0, 0
            average_acc = 0

            # Declare model
            model = FFNN()
            if cuda_flag:
                model = model.cuda()
                loss_function = loss_function.cuda()
            optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(model)

            for i, (batch_signal, batch_target) in enumerate(trainloader):
                s = variable(batch_signal).float()# (batch size, sequence length)
                t = variable(batch_target)
                optimizer.zero_grad()
                model.train()
                pred = model(s)
                loss = loss_function(pred, t)
                loss.backward()
                optimizer.step()
                for l, p in zip(t.data.cpu().numpy(), pred.data.cpu().numpy()):
                    max_idx = np.argmax(p)
                    pcount[max_idx] += 1
                    if max_idx == l:
                        total_acc += 1
                total_loss += loss.item()
                average_loss = total_loss / (i+1)
                average_acc = total_acc / (i+1) # 50/200 = 0.25
                print('\rTraining: {}/{} loss:{:.4f} acc:{:.4f}'.format(i+1, len(trainloader), average_loss, average_acc), end='')
            '''
            if average_acc > 0.93:
            # save it
                best_model = copy.deepcopy(model)
                torch.save(best_model.state_dict(), '{}/params_{}_best_epoch{}_acc{:.4f}_loss{:.4f}_drop{}'.format(MD_SAVE_DIR, DATE, epoch, average_acc, average_loss, dropout))
                break
            '''
            #------validation------
            print("start validation")
            model.eval()
            valid_total_loss, valid_total_acc= 0.0, 0
            v_s = variable(x_val).float()
            v_t = variable(y_val)
            v_pred = model(v_s)
            v_loss =  loss_function(v_pred, v_t)
            for l, p in zip(v_t.data.cpu().numpy(), v_pred.data.cpu().numpy()):
                 if np.argmax(p) == l:
                     valid_total_acc += 1
            valid_total_loss +=  v_loss.item()
            valid_average_acc[fold_idx] = valid_total_acc / len(y_val)
            valid_average_loss[fold_idx] = valid_total_loss / len(y_val)
            print('Fold{}: loss:{:.4f} acc:{:.4f} valid_loss:{:.4f} valid_acc:{:.4f}\n'.format(fold_idx+1, average_loss, average_acc, valid_average_loss[fold_idx], valid_average_acc[fold_idx]), end='')
            fold_idx += 1
        history_valid_loss.append(sum(valid_average_loss)/len(valid_average_loss))
        history_valid_acc.append(sum(valid_average_acc)/len(valid_average_acc))
        print('10-Fold average: valid_loss:{:.4f} valid_acc:{:.4f}\n'.format(sum(valid_average_loss)/len(valid_average_loss), sum(valid_average_acc)/len(valid_average_acc)), end='')
    for i, minibatch in enumerate(minibatchs):
        print("mini-batch size:", minibatch)
        print("loss:", history_valid_loss[i])
        print("acc:", history_valid_acc[i])

'''
    print("preparing training dataset...")
    signal, target = [], []
    for data in test_data:
        signal.append(data[0])
        target.append(data[1])
    
    print("\nstart testing...")
    #------testing------
    best_model.eval()
    test_total_loss, test_total_acc = 0.0, 0
    count = 0
    test_signal = variable(signal).float()
    test_target = variable(target)
    test_pred = best_model(signal)
    test_loss =  loss_function(test_pred, test_target)
    for l, p in zip(test_target.data.cpu().numpy(), test_pred.data.cpu().numpy()):
        if np.argmax(p) == l:
            test_total_acc += 1
    test_total_loss +=  test_loss.item()
    count += len(test_pred)
    test_average_acc = test_total_acc / count
    test_average_loss = test_total_loss / len(testloader)
    print('\rtesting loss:{:.4f} acc:{:.4f}'.format(test_average_loss, test_average_acc), end='')
'''
