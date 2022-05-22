import os
import time
import pickle
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime
import pandas as pd
from tool import EarlyStopping
from sklearn.metrics import roc_auc_score,mean_squared_error

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler, TensorDataset
from torch.utils.data.sampler import RandomSampler
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')


# save dict    
def save_obj(obj,name):
    with open(name + '.pkl','wb') as f:
        pickle.dump(obj,f)
                  
#load dict        
def load_obj(name):
    with open(name +'.pkl','rb') as f:
        return pickle.load(f)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def interp(v, q, num):
    f = interpolate.interp1d(v,q,kind='linear')
    v_new = np.linspace(v[0],v[-1],num)
    q_new = f(v_new)
    vq_new = np.concatenate((v_new.reshape(-1,1),q_new.reshape(-1,1)),axis=1)
    return q_new

def get_xy(name, n_cyc, in_stride, fea_num, v_low, v_upp, q_low, q_upp, rul_factor, cap_factor):
    """
    Args:
        n_cyc (int): The previous cycles number for model input
        in_stride (int): The interval in the previous cycles number
        fea_num (int): The number of interpolation
        v_low (float): Voltage minimum for normalization
        v_upp (float): Voltage maximum for normalization
        q_low (float): Capacity minimum for normalization
        q_upp (float): Capacity maximum for normalization
        rul_factor (float): The RUL factor for normalization
        cap_factor (float): The capacity factor for normalization
    """
    A = load_obj(f'./data/our_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']

    all_idx = list(A_dq.keys())[9:]
    all_fea, rul_lbl, cap_lbl = [], [], []
    for cyc in all_idx:
        tmp = A_df[cyc]
        tmp = tmp.loc[tmp['Status'].apply(lambda x: not 'discharge' in x)]
        
        left = (tmp['Current (mA)']<5000).argmax() + 1
        right = (tmp['Current (mA)']<1090).argmax() - 2

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Voltage (V)'].values
        tmp_q = tmp['Capacity (mAh)'].values
        tmp_t = tmp['Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1,1), q_fea.reshape(-1,1)))
        
        all_fea.append(np.expand_dims(tmp_fea,axis=0))
        rul_lbl.append(A_rul[cyc])
        cap_lbl.append(A_dq[cyc])
    all_fea = np.vstack(all_fea)
    rul_lbl = np.array(rul_lbl)
    cap_lbl = np.array(cap_lbl)
    
    all_fea_c = all_fea.copy()
    all_fea_c[:,:,0] = (all_fea_c[:,:,0]-v_low)/(v_upp-v_low)
    all_fea_c[:,:,1] = (all_fea_c[:,:,1]-q_low)/(q_upp-q_low)
    dif_fea = all_fea_c - all_fea_c[0:1,:,:]
    all_fea = np.concatenate((all_fea,dif_fea),axis=2)
    
    all_fea = np.lib.stride_tricks.sliding_window_view(all_fea,(n_cyc,fea_num,4))
    cap_lbl = np.lib.stride_tricks.sliding_window_view(cap_lbl,(n_cyc,))
    all_fea = all_fea.squeeze(axis=(1,2,))
    rul_lbl = rul_lbl[n_cyc-1:]
    all_fea = all_fea[:,(in_stride - 1)::in_stride,:,:]
    cap_lbl = cap_lbl[:,(in_stride - 1)::in_stride,]
    
    all_fea_new = np.zeros(all_fea.shape)
    all_fea_new[:,:,:,0] = (all_fea[:,:,:,0]-v_low)/(v_upp-v_low)
    all_fea_new[:,:,:,1] = (all_fea[:,:,:,1]-q_low)/(q_upp-q_low)
    all_fea_new[:,:,:,2] = all_fea[:,:,:,2]
    all_fea_new[:,:,:,3] = all_fea[:,:,:,3]
    print(f'{name} length is {all_fea_new.shape[0]}', 
          'v_max:', '%.4f'%all_fea_new[:,:,:,0].max(),
          'q_max:', '%.4f'%all_fea_new[:,:,:,1].max(),
          'dv_max:', '%.4f'%all_fea_new[:,:,:,2].max(), 
          'dq_max:', '%.4f'%all_fea_new[:,:,:,3].max())
    rul_lbl = rul_lbl / rul_factor
    cap_lbl = cap_lbl / cap_factor
    
    return all_fea_new,np.hstack((rul_lbl.reshape(-1,1),cap_lbl))


class Trainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss
            alpha (List: [float]): The weights of Capacity loss
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.alpha = alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        alpha = self.alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)
            
            # validate
            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)

                    y_pred.append(y_)
                    y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            if self.n_epochs > 100:
                if (epoch % 100 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',f'-- "total" loss {losses:.4}')

            early_stopping(epoch_loss, model, f'{model_name}_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_end.pt')

        return model, train_loss, valid_loss, total_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred
    

class FineTrainer():
    
    def __init__(self, lr, n_epochs,device, patience, lamda, train_alpha, valid_alpha, model_name):
        """
        Args:
            lr (float): Learning rate
            n_epochs (int): The number of training epoch
            device: 'cuda' or 'cpu'
            patience (int): How long to wait after last time validation loss improved.
            lamda (float): The weight of RUL loss. In fine-tuning part, set 0.
            train_alpha (List: [float]): The weights of Capacity loss in model training
            valid_alpha (List: [float]): The weights of Capacity loss in model validation
            model_name (str): The model save path
        """
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.model_name = model_name
        self.lamda = lamda
        self.train_alpha = train_alpha
        self.valid_alpha = valid_alpha

    def train(self, train_loader, valid_loader, model, load_model):
        model = model.to(self.device)
        device = self.device
        optimizer = optim.Adam(model.parameters(), lr=self.lr,)
        model_name = self.model_name
        lamda = self.lamda
        train_alpha = self.train_alpha
        valid_alpha = self.valid_alpha
        
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(self.patience, verbose=True)
        loss_fn.to(self.device)

        # Training
        train_loss = []
        valid_loss = []
        total_loss = []
        added_loss = []
        
        for epoch in range(self.n_epochs):
            model.train()
            y_true, y_pred = [], []
            losses = []
            for step, (x,y) in enumerate(train_loader):  
                optimizer.zero_grad()
                
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)
                
                loss = lamda * loss_fn(y_.squeeze(), y[:,0])
                
                for i in range(y.shape[1] - 1):
                    loss += loss_fn(soh_[:,i], y[:,i+1]) * train_alpha[i]
                    
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().numpy())

                y_pred.append(y_)
                y_true.append(y[:,0])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            train_loss.append(epoch_loss)
            
            losses = np.mean(losses)
            total_loss.append(losses)

            # validate
            model.eval()
            y_true, y_pred, all_true, all_pred = [], [], [], []
            with torch.no_grad():
                for step, (x,y) in enumerate(valid_loader):
                    x = x.to(device)
                    y = y.to(device)
                    y_, soh_ = model(x)
                    soh_ = soh_.view(y_.shape[0], -1)

                    y_pred.append(y_)
                    y_true.append(y[:,0])
                    all_true.append(y[:,1:])
                    all_pred.append(soh_)

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            all_true = torch.cat(all_true, axis=0)
            all_pred = torch.cat(all_pred, axis=0)
            epoch_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
            valid_loss.append(epoch_loss)
            
            temp = 0
            for i in range(all_true.shape[1]):
                temp += mean_squared_error(all_true[0:1,i].cpu().detach().numpy(), 
                                           all_pred[0:1,i].cpu().detach().numpy()) * valid_alpha[i]
            added_loss.append(temp)
            
            if self.n_epochs > 10:
                if (epoch % 200 == 0 and epoch !=0):
                    print('Epoch number : ', epoch)
                    print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                          f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')
            else :
                print(f'-- "train" loss {train_loss[-1]:.4}', f'-- "valid" loss {epoch_loss:.4}',
                      f'-- "total" loss {losses:.4}',f'-- "added" loss {temp:.4}')

            early_stopping(temp, model, f'{model_name}_fine_best.pt')
            if early_stopping.early_stop:
                break
                
        if load_model:
            model.load_state_dict(torch.load(f'{model_name}_fine_best.pt'))
        else:
            torch.save(model.state_dict(), f'{model_name}_fine_end.pt')

        return model, train_loss, valid_loss, total_loss, added_loss

    def test(self, test_loader, model):
        model = model.to(self.device)
        device = self.device

        y_true, y_pred, soh_true, soh_pred = [], [], [], []
        model.eval()
        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)
                y_, soh_ = model(x)
                soh_ = soh_.view(y_.shape[0], -1)

                y_pred.append(y_)
                y_true.append(y[:,0])
                soh_pred.append(soh_)
                soh_true.append(y[:,1:])

            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)
            soh_true = torch.cat(soh_true, axis=0)
            soh_pred = torch.cat(soh_pred, axis=0)
            mse_loss = mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        return y_true, y_pred, mse_loss, soh_true, soh_pred