import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import os
import pickle
import re
import xlrd
import tqdm

from scipy import interpolate
from copy import deepcopy
from scipy import stats
from scipy.optimize import leastsq
from scipy.stats import pearsonr

from common import *

import warnings
warnings.filterwarnings("ignore")

data_path = './data/nmc_data/original/'
files = os.listdir(data_path)
files = [file for file in files if 'SNL' in file]
bat_prefix = list(set([i[:-15] for i in files]))

bat_prefix = tqdm.tqdm(bat_prefix)
for prefix in bat_prefix:
    cyc_v = {}
    cyc_rul = {}
    cyc_dq  = {}
    
    cycle_df = prefix + '_cycle_data.csv'
    time_df  = prefix + '_timeseries.csv'
    cycle_df = pd.read_csv(data_path + cycle_df)
    time_df  = pd.read_csv(data_path + time_df)
    
    tmp = cycle_df[['Cycle_Index','Discharge_Capacity (Ah)']]
    init_cap = tmp['Discharge_Capacity (Ah)'].iloc[0]
    end_cap = init_cap * 0.8
    tmp = tmp[tmp['Discharge_Capacity (Ah)'] < end_cap]
    tmp = tmp.Cycle_Index.values
    for i in range(len(tmp) - 1, 0, -1):
        if tmp[i] - tmp[i - 1] != 1:
            break
    life_cyc = tmp[i]
    
    cyc_list = []
    for i in range(len(cycle_df)):
        if 0 < i < len(cycle_df) - 1:
            if abs(cycle_df.iloc[i]['Discharge_Capacity (Ah)'] - cycle_df.iloc[i - 1]['Discharge_Capacity (Ah)']) >= 0.05:
                continue
            if abs(cycle_df.iloc[i]['Discharge_Capacity (Ah)'] - cycle_df.iloc[i + 1]['Discharge_Capacity (Ah)']) >= 0.05:
                continue      
        tmp = cycle_df.iloc[i]
        if 1 <= tmp.Cycle_Index < life_cyc and end_cap <= tmp['Discharge_Capacity (Ah)']<= init_cap:
            cyc = tmp.Cycle_Index
            cyc_list.append(cyc) 
            cyc_rul.update({cyc:life_cyc-cyc})
            cyc_dq.update({cyc:tmp['Discharge_Capacity (Ah)']})
    for cyc in cyc_list:
        tmp = time_df[time_df.Cycle_Index == cyc]
        tmp = tmp.reset_index(drop=True)
        tmp['Test_Time (s)'] = tmp['Test_Time (s)'] - tmp['Test_Time (s)'].iloc[0]
        cyc_v.update({cyc: tmp})
    
    bats_dic = {}
    bats_dic.update({prefix:{'rul':cyc_rul,
                    'dq':cyc_dq,
                    'data':cyc_v}})
    save_obj(bats_dic,'./data/nmc_data/'+prefix)
    
pkl_list = os.listdir('./data/nmc_data/')
pkl_list = [i for i in pkl_list if 'SNL' in i]

train_name = []
for name in pkl_list:
    train_name.append(name[:-4])
    
def get_xy(name):
    A = load_obj(f'./data/nmc_data/{name}')[name]
    A_rul = A['rul']
    A_dq = A['dq']
    A_df = A['data']

    all_idx = list(A_dq.keys())[9:]
    all_fea, all_lbl, aux_lbl = [], [], []
    for cyc in all_idx:
        tmp = A_df[cyc]

        init_cap = tmp['Charge_Capacity (Ah)'].iloc[-1] * 0.8
        left = (tmp['Charge_Capacity (Ah)'] > init_cap).argmax() - 20

    #     left = (tmp['电流(mA)']<5000).argmax() + 1

        current = tmp['Current (A)'].values
        for i in range(len(current)):
            if current[i] > 0:
                break
        i += 1
        pos = np.where(current < current[i])[0]
        for j in pos:
            if j > i:
                break
        right = j + 20

        if left >= right - 1:
            continue

        tmp = tmp.iloc[left:right]

        tmp_v = tmp['Voltage (V)'].values
        tmp_q = tmp['Charge_Capacity (Ah)'].values
        tmp_t = tmp['Test_Time (s)'].values
        v_fea = interp(tmp_t, tmp_v, fea_num)
        q_fea = interp(tmp_t, tmp_q, fea_num)

        tmp_fea = np.hstack((v_fea.reshape(-1,1), q_fea.reshape(-1,1)))

        all_fea.append(np.expand_dims(tmp_fea,axis=0))
        all_lbl.append(A_rul[cyc])
        aux_lbl.append(A_dq[cyc])
#     print(len(all_fea))
    all_fea = np.vstack(all_fea)
    all_lbl = np.array(all_lbl)
    aux_lbl = np.array(aux_lbl)
    
    all_fea_c = all_fea.copy()
    all_fea_c[:,:,0] = (all_fea_c[:,:,0]-v_low)/(v_upp-v_low)
    all_fea_c[:,:,1] = (all_fea_c[:,:,1]-q_low)/(q_upp-q_low)
    dif_fea = all_fea_c - all_fea_c[0:1,:,:]
    all_fea = np.concatenate((all_fea,dif_fea),axis=2)
#     print(all_fea.shape,all_lbl.shape)
    
#     print(all_fea.shape)
    all_fea = np.lib.stride_tricks.sliding_window_view(all_fea,(n_cyc,fea_num,4))
    aux_lbl = np.lib.stride_tricks.sliding_window_view(aux_lbl,(n_cyc,))
#     print(all_fea.shape)
    all_fea = all_fea.squeeze(axis=(1,2,))
#     print(all_fea.shape)
    all_lbl = all_lbl[n_cyc-1:]
    all_fea = all_fea[::stride]
    all_fea = all_fea[:,::in_stride,:,:]
    all_lbl = all_lbl[::stride]
    aux_lbl = aux_lbl[::stride]
    aux_lbl = aux_lbl[:,::in_stride,]
    
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
    all_lbl = all_lbl / lbl_factor
    aux_lbl = aux_lbl / aux_factor
    
    return all_fea_new,np.hstack((all_lbl.reshape(-1,1),aux_lbl))

n_cyc = 30
in_stride = 3
fea_num = 100

v_low = 3
v_upp = 4.3
q_low = 1.2
q_upp = 2.9
lbl_factor = 2000
aux_factor = 2.9

stride = 1
all_loader = dict()
all_fea = []
all_lbl = []
print('----init_train----')
for name in train_name:
    tmp_fea, tmp_lbl = get_xy(name)
    all_loader.update({name:{'fea':tmp_fea,'lbl':tmp_lbl}})
    all_fea.append(tmp_fea)
    all_lbl.append(tmp_lbl)
save_obj(all_loader,'./data/nmc_data/nmc_loader')