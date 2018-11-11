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
#from tokenizer import *
from tensorboardX import SummaryWriter
from random import shuffle
import time
from sklearn.model_selection import KFold


global_step = 0
DATE = "1031" # dropout 0.5
out_size = 3
#minibatch = 100
lr = 0.01 #0.01
max_epoch = 500
dropout = 0#0.5 #0.1
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', action='store', type=int, default='0')
    parser.add_argument('-batch_size', action='store', type=int, default='1')
    parser.add_argument('-save_path', action='store', default='./runs/')
    parser.add_argument('-save', action='store_true', default=False)
    args = parser.parse_args()
    if args.device < 0:
        cuda_flag = False
        print("Only using CPU")
    else:
        torch.cuda.set_device(args.device)
        print("Current GPU device:", torch.cuda.current_device())
    print("Current dropout rate:", dropout)
    
    minibatch = args.batch_size
    print("Current batch size:", minibatch)
    # Create current model directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    timestep = str(int(time.time()))

    if args.save:
        os.makedirs(os.path.join(args.save_path, timestep))
        os.makedirs(os.path.join(args.save_path, timestep, 'model'))
        os.makedirs(os.path.join(args.save_path, timestep, 'log'))
        MD_SAVE_DIR = os.path.abspath(os.path.join(args.save_path, timestep, 'model')) # runs/[timestep]/model/
        print(f'----------Making Directory at {timestep}--------')

    # Create SummaryWriter
    #writer = SummaryWriter(log_dir=os.path.join(args.save_path, timestep, 'log'))

    # File Path
    # -load-
    DATA           = 'data/np_data.pkl'
    

    # Declare model
    model = FFNN()

    if cuda_flag:
        model = model.cuda()
        loss_function = loss_function.cuda()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(model)

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
    print(train_data)
    test_data = list(itertools.chain(n_test_data, p_test_data))
    print(test_data)
    shuffle(train_data)
    shuffle(test_data)
    
    # Prepare training & testing dataset
    print("preparing training dataset...")
    signal, target = [], []
    for data in train_data:
        signal.append(data[0])
        target.append(data[1])
    train_dataset = make_data.myDataset(signal=signal, target=target)
    trainloader =  Data.DataLoader(dataset=train_dataset, batch_size=minibatch, shuffle=True, collate_fn=make_data.my_collate, num_workers=0)
    print("length of trainloader:", len(trainloader))

    print("preparing testing dataset..")
    t_signal, t_target = [], []
    for data in test_data:
        t_signal.append(data[0])
        t_target.append(data[1])
    test_dataset = make_data.myDataset(signal=t_signal, target=t_target)
    testloader =  Data.DataLoader(dataset=test_dataset, batch_size=minibatch, shuffle=True, collate_fn=make_data.my_collate, num_workers=0)
    print("length of testloader:", len(testloader))

    print("start training...")
    # ------training------
    for epoch in range(0, max_epoch):
        start_time = time.time()
        count, pcount = [0]*out_size, [0]*out_size
        total_loss, total_acc = 0.0, 0
        average_acc = 0
        for i, (batch_signal, batch_target) in enumerate(trainloader):
            s = variable(batch_signal).float()# (batch size, sequence length)
            #print(s)
            t = variable(batch_target)
            #print(t)
            #input("pause")
            optimizer.zero_grad()
            model.train()
            pred = model(s)
            #print(pred)
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
            print('\repoch:{} {}/{} loss:{:.4f} acc:{:.4f}'.format(epoch+1, i+1, len(trainloader), average_loss, average_acc), end='')
            #writer.add_scalar('data/train_loss', average_loss, global_step)
            #writer.add_scalar('data/train_acc', average_acc, global_step)
            global_step += 1
        
        if average_acc > 0.93:
        # save it
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), '{}/params_{}_best_epoch{}_acc{:.4f}_loss{:.4f}_drop{}'.format(MD_SAVE_DIR, DATE, epoch, average_acc, average_loss, dropout))
            break

    print("\nstart testing...")
    #------testing------
    best_model.eval()
    test_total_loss, test_total_acc = 0.0, 0
    count = 0
    for i, (batch_signal, tcbatch_target) in enumerate(testloader):
        test_signal = variable(batch_signal).float()
        test_target = variable(batch_target)
        test_pred = best_model(test_signal)
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
        print("\nstart testing...")
        #------validation------
        model.eval()
        valid_total_loss, valid_total_acc = 0.0, 0
        count = 0
        for i, (batch_premise, batch_hypothesis, batch_target) in enumerate(valloader):
            v1 = variable(batch_premise)
            v2 = variable(batch_hypothesis)
            v_target = variable(batch_target)
            v_pred = model(v1, v2)
            v_loss =  loss_function(v_pred, v_target)
            #print(v_loss)
            for l, p in zip(v_target.data.cpu().numpy(), v_pred.data.cpu().numpy()):
                 if np.argmax(p) == l:
                     valid_total_acc += 1
            valid_total_loss +=  v_loss.item()
            count += len(v_pred)
        valid_average_acc = valid_total_acc / count
        valid_average_loss = valid_total_loss / len(valloader)
        print('epoch:{} loss:{:.4f} acc:{:.4f} valid_loss:{:.4f} valid_acc:{:.4f}\n'.format(epoch+1, average_loss, average_acc, valid_average_loss, valid_average_acc), end='')
        writer.add_scalar('data/val_loss', valid_average_loss, global_step)
        writer.add_scalar('data/val_acc', valid_average_acc, global_step)

        #---earlystop---
        if valid_average_loss + 0.0001 >= min(history_valid_loss):
            earlystop_counter += 1
        # if find a better model
        else:
            history_valid_loss.append(valid_average_loss)
            earlystop_counter = 0
            best_model = copy.deepcopy(model)
            best_train_info[0] = average_acc
            best_train_info[1] = average_loss
            best_dev_info[0] = valid_average_acc
            best_dev_info[1] = valid_average_loss
        # save it
            try:
                torch.save(best_model.state_dict(), '{}/hw3_params_{}_best_epoch{}_acc{:.4f}_loss{:.4f}_drop{}'.format(MD_SAVE_DIR, DATE, epoch, average_acc, average_loss, dropout))
            except:
                pass
        print("earlystop_counter:", earlystop_counter)
        if earlystop_counter >= earlystop_threshold:
            print("---Training Setting---")
            print("Dropout:", dropout, )
            print("Learning rate:", lr)
            print("L2 penalty:", weight_decay)
            print("Earlystop threshold:", earlystop_threshold)
            print("---Info of Best Model---")
            print("Train acc:{:.4f}/1oss:{:.4f}".format(best_train_info[0], best_train_info[1]), end='\n' )
            print("Dev acc:{:.4f}/loss:{:.4f}".format(best_dev_info[0], best_dev_info[1]), end='\n' )
            break
        print("Training time of epoch{}: {} sec".format(epoch, time.time()-start_time))

    # Close writer
    writer.close()
'''
