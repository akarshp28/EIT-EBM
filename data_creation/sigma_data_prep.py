#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[ ]:


def ReadData(fileName):
    Dict = scipy.io.loadmat(fileName)
    return Dict

def ExtractX(Dict, key1, key2):
    YOMEG = Dict[key1]
    XOMEG = Dict[key2]
    Xall = np.zeros((XOMEG.shape[0], 2), dtype=np.int64)
    Xall[:, 0] = np.transpose(XOMEG)
    Xall[:, 1] = np.transpose(YOMEG)
    return Xall

def get_2d(Xall, data):
    all_pts = np.zeros((len(Xall)), dtype=np.float64)
    for i in range(Xall.shape[0]):
        x = Xall[i, 0] - 1
        y = Xall[i, 1] - 1
        all_pts[i] = data[y, x]
    return all_pts

def SetX(Xall, CorrectB, CorrectS, inputSize=2):
    Xnor = np.zeros([Xall.shape[0], inputSize])
    Xnor[:,0] = (Xall[:,0]-CorrectB)/CorrectS
    Xnor[:,1] = (Xall[:,1]-CorrectB)/CorrectS
    return Xnor


# In[ ]:


data_2d = True # if 2D data else image 128x128
ebm_data = False  # yes/no labels
circle_data = False # apply mask


# In[ ]:


plot_data = False
folder_names = ['InputData'+str(i) for i in range(9)]
combined_data = []
label_data = []


# In[ ]:


if data_2d:
    for f in folder_names:
        path = './Data/' + f + '/'
        print(path, flush=True)
        
        for ind, file in enumerate(os.listdir(path)):
            if file.endswith(".mat"):
                
                Dict = ReadData(path + file)
                
                if 'Xall' in Dict.keys():
                    XY_all = ExtractX(Dict, 'Yall', 'Xall')
                
                XY_circle = ExtractX(Dict, 'YOmeg', 'XOmeg')
                sigma_2d = get_2d(XY_circle, Dict['Sigma'])
                combined_data.append(sigma_2d)

else:
    
    for f in folder_names:
        path = './Data/' + f + '/'
        print(path, flush=True)
        
        for ind, file in enumerate(os.listdir(path)):
            if file.endswith(".mat"):
                
                Dict = ReadData(path + file)
                
                if circle_data:
                    sigma = Dict['Sigma'] * Dict['Omega']
                else:
                    sigma = Dict['Sigma']
                
                if ebm_data:
                    combined_data.append(sigma)
                else:
                    if len(Dict['anomaly_list'][0]) == 18:
                        label = [0.0, 0.0, 1.0] # 3
                    elif len(Dict['anomaly_list'][0]) == 12:
                        label = [0.0, 1.0, 0.0] # 2
                    else:
                        label = [1.0, 0.0, 0.0] # 1
                    
                    combined_data.append(sigma)
                    label_data.append(label)
                
                if plot_data:
                    plt.figure(figsize=(7,6))
                    plt.colorbar(plt.imshow(sigma))
                    plt.title('Sigma')
                    plt.tight_layout()
                    plt.show()


# In[ ]:


if ebm_data:
    combined_data = np.array(combined_data)
    print(combined_data.shape, flush=True)
    
    xtrain, xtest = train_test_split(combined_data, test_size=0.2)
    xtrain, xval = train_test_split(xtrain, test_size=0.2)
    print('X: ', xtrain.shape, xval.shape, xtest.shape, flush=True)
    
    combo_split = {'xtrain': xtrain, 'xval': xval, 'xtest': xtest}
    
    if circle_data:
        with open('multisigma_ebm_128.pickle', 'wb') as f:
            pickle.dump(combo_split, f)
    else:
        with open('multisigma_ebm_128_no_circle.pickle', 'wb') as f:
            pickle.dump(combo_split, f)

elif data_2d:
    combined_data = np.array(combined_data)
    print(combined_data.shape, flush=True)
    
    xtrain, xtest = train_test_split(combined_data, test_size=0.2)
    xtrain, xval = train_test_split(xtrain, test_size=0.2)
    print('X: ', xtrain.shape, xval.shape, xtest.shape, flush=True)
    
    combo_split = {'xtrain': xtrain, 'xval': xval, 'xtest': xtest, 
                   'XY_all': XY_all, 'Xall': XY_circle}
    
    with open('multisigma_ebm_128_2D.pickle', 'wb') as f:
        pickle.dump(combo_split, f)

else:
    combined_data = np.array(combined_data)
    label_data = np.array(label_data)
    print(combined_data.shape, label_data.shape, flush=True)
    
    xtrain, xtest, ytrain, ytest = train_test_split(combined_data, label_data, test_size=0.2)
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2)
    print('X: ', xtrain.shape, xval.shape, xtest.shape, flush=True)
    print('Y: ', ytrain.shape, yval.shape, ytest.shape, flush=True)
    
    combo_split = {'xtrain': xtrain, 'xval': xval, 'xtest': xtest,
                   'ytrain': ytrain, 'yval': yval, 'ytest': ytest}
    
    if circle_data:
        with open('multisigma_classify_128.pickle', 'wb') as f:
            pickle.dump(combo_split, f)
    else:
        with open('multisigma_classify_128_no_circle.pickle', 'wb') as f:
            pickle.dump(combo_split, f)

print('Data saved', flush=True)

