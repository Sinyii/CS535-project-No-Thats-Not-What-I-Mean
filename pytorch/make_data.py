import os
import csv
import json
import pickle

# This part will be called by train.py
class myDataset(Data.Dataset):
    def __init__(self, signal, target):
        self.signal = signal
        self.target = target


    def __getitem__(self, index):
        return_signal, return_target = self.signal[index], self.target[index]
        return return_signal, return_target

    def __len__(self):
        return len(self.signal)

def my_collate(batch):
    signal, target = [], []
    for item in batch:
        signal.append(item[0])
        target.append(item[1])
    return [signal, target]
###

SAVE_PATH = './data/np_data.pkl'
paths = ['./subject1/', './subject2/','./subject3/']

data_count = 0
files_count = 0
datas = []
n, p = [], [] 
for path in paths:
    dirs = os.listdir(path)
    for f in dirs:
        with open(path + f, newline='') as csvfile:
            data=[]
            target = 0
            rows = csv.reader(csvfile, delimiter=':')
            for row in rows: # process data
                r = row[0].split(',')
                data.append(int(r[0]))

            tmp = f.split('.')
            file_name = tmp[0]
            mid = len(file_name)/2
            if mid.is_integer() == False:
                target = 1 # p300 wave
            else:
                if file_name[:int(mid)] == file_name[int(mid):]:
                    target = 0 #normal wave
                else:
                    target = 1 # p300 wave
            datas.append([data, target])
            if target == 0:
                n.append([data, target])
                print(f, "Normal")
            else:
                p.append([data, target])
                print(f, "p300")
            data_count += len(data)
    files_count += len(dirs)

print("Avg. data length:", data_count/files_count)
print("Total data:", files_count)    
print("Normal:", len(n))
print("P300:", len(p))

with open(SAVE_PATH, 'wb') as f:
    pickle.dump((n,p), f)
    print('\rdata current position(=data size):{}\n'.format(f.tell()), end='')

