# Health_status_prediction
This is a PyTorch implementation of the paper: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning. 
Ye Yuan, Guijun Ma, Songpei Xu
## Requirements
This model is implemented using Python3 with dependencies specified in requirements.txt
```
pip install -r requirements.txt
```
## Data Preparation
[Data Download](https://doi.org/10.17632/nsc7hnsg4s.2)  
Yuan, Ye; Ma, Guijun; Xu, Songpei (2022), “The Dataset for: Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning ”, Mendeley Data, V2, doi: 10.17632/nsc7hnsg4s.2
## Code Introduction
- [tool.py](https://github.com/HAIRLAB/Health_status_prediction/blob/main/tool.py) : Early stopping function
- [common.py](https://github.com/HAIRLAB/Health_status_prediction/blob/main/common.py) : Including data preprocessing, model training and validation
- [net.py](https://github.com/HAIRLAB/Health_status_prediction/blob/main/net.py) : Model structure  
- [prepare_ne_data.py](https://github.com/HAIRLAB/Health_status_prediction/blob/main/prepare_ne_data.py) : Data preprocessing for Task B  
- [prepare_nmc_data.py](https://github.com/HAIRLAB/Health_status_prediction/blob/main/prepare_nmc_data.py) : Data preprocessing for Task C
- [1-wx_inner.ipynb](https://github.com/HAIRLAB/Health_status_prediction/blob/main/1-wx_inner.ipynb) : The pipeline of the Task A  
- [2-ne2wx.ipynb](https://github.com/HAIRLAB/Health_status_prediction/blob/main/2-ne2wx.ipynb) : The pipeline of the Task B  
- [3-nmc2lfp.ipynb](https://github.com/HAIRLAB/Health_status_prediction/blob/main/3-nmc2lfp.ipynb) : The pipeline of the Task C

