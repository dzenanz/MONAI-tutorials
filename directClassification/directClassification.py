import logging
import os
import sys
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


model_path = os.getcwd() + "/miqaIQM.pth"
torch.manual_seed(1983)

def parseJSON(jsonFile, scanroot, features, decisions, filenames):
    with open(jsonFile) as json_file:
        data = json.load(json_file)
        for s in data['scans']:
            decision = int(s['decision'])
            decLen = len(decisions)
            if (decLen > 1 and decisions[decLen - 1] == 1 and decision == 0):
                print("zero at index", decLen)
            path = scanroot + data['data_root'] + s['path'] + "/"
            for k, v in s['volumes'].items():
                del v['warnings']
                del v['output']
                del v['Patient']
                v['VRX'] = float(v['VRX'])
                v['VRY'] = float(v['VRY'])
                v['VRZ'] = float(v['VRZ'])

                vl = list(v.values())
                allFinite = True
                for fv in vl:
                    if (not math.isfinite(fv)):
                        allFinite = False

                if (allFinite):
                    filenames.append(path + k + ".nii.gz")
                    decisions.append(decision)
                    features.append(vl)
                else:
                    print("Non-finite value encountered, timestep skipped")

def main():
    # monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    filenames = []
    features = []
    decisions = []
    parseJSON(os.getcwd() + '/SRI_Sessions/session_ann_pretty.json',
              os.getcwd() + "/SRI_Sessions/scanroot",
              features, decisions, filenames)

    # print(decisions)
    print("timepoint count:", len(features))

    # 2 binary labels for scan classification: 1=good, 0=bad
    y = np.asarray(decisions, dtype=np.int64)
    X = np.asarray(features, dtype=np.float32)



    # rest is based on
    # https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2300, shuffle=False)

    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.fit_transform(X_test)

    class trainData(Dataset):    
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data
            
        def __getitem__(self, index):
            return self.X_data[index], self.y_data[index]
            
        def __len__(self):
            return len(self.X_data)

    class testData(Dataset):    
        def __init__(self, X_data):
            self.X_data = X_data
            
        def __getitem__(self, index):
            return self.X_data[index]
            
        def __len__(self):
            return len(self.X_data)

    test_data = testData(torch.FloatTensor(X_test))
    train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)

    class binaryClassification(nn.Module):
        def __init__(self):
            super(binaryClassification, self).__init__()
            # Number of input features is 21
            self.layer_1 = nn.Linear(21, 64)
            self.layer_2 = nn.Linear(64, 64)
            self.layer_out = nn.Linear(64, 1) 
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(64)
            self.batchnorm2 = nn.BatchNorm1d(64)
            
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)
            
            return x

    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        
        return acc

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = binaryClassification()
    #if (os.path.exists(model_path)):
    #    model.load_state_dict(torch.load(model_path))
    #    print(f"Loaded NN model from file '{model_path}'")
    #else:
    #    print("Training NN from scratch")
    model.to(device)
    print(model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for e in range(0, 10):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    # save the current model
    torch.save(model.state_dict(), model_path)

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    print("confusion_matrix:")
    print(confusion_matrix(y_test, y_pred_list))
    print(classification_report(y_test, y_pred_list))

if __name__ == "__main__":
    main()