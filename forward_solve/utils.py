#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import csv
import copy
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

# In[ ]:

def reset_weights(model):
  for layer in model.layers: 
    if isinstance(layer, tf.keras.Model):
      reset_weights(layer)
      continue
    for k, initializer in layer.__dict__.items():
      if "initializer" not in k:
        continue
      # find the corresponding variable
      var = getattr(layer, k.replace("_initializer", ""))
      var.assign(initializer(var.shape, var.dtype))


def mesh_free(MaxNumOfPoints, Dict, keys, main_key, meshfree):
    tempVar = np.zeros((MaxNumOfPoints, 1), dtype=np.float64)
    aVec = []
    AVec = []
    D = {}
    for i in range(len(keys)):
        a = copy.deepcopy(tempVar)
        aVec.append(a)
        curKey = keys[i]
        AVec.append(Dict[curKey])
    
    if meshfree:
        XallRand = Dict[main_key][:MaxNumOfPoints]
    else:
        XallRand = Dict[main_key]
    for i in range(MaxNumOfPoints):
        y = XallRand[i, 1] - 1
        x = XallRand[i, 0] - 1
        
        for k in range(len(keys)):
            A = AVec[k]
            aVec[k][i] = A[y,x]
    
    for k in range(len(keys)):
        a = aVec[k]
        a = np.squeeze(a)
        a = np.reshape(a, [XallRand.shape[0], 1])
        D.update({keys[k]: a})
    
    if meshfree:
        XallRandnor = SetX(XallRand, Dict['CorrectB'], Dict['CorrectS'])
        return XallRandnor, D
    else:
        return D

def unsup_mlp(model_name):
    xy_input_layer = tf.keras.layers.Input(shape=(2))
    x = tf.keras.layers.Dense(26, activation='tanh')(xy_input_layer)
    x = tf.keras.layers.Dense(26, activation='tanh')(x)
    x = tf.keras.layers.Dense(26, activation='tanh')(x)
    x = tf.keras.layers.Dense(10, activation='tanh')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.models.Model(xy_input_layer, x, name=model_name)
    return model

def mlp(units, Numlayers, out_dim, model_name):
    if model_name == 'SNET':
        input_layer = tf.keras.layers.Input(shape=(2))
    else:
        xy_input_layer = tf.keras.layers.Input(shape=(2))
        data_input_layer = tf.keras.layers.Input(shape=(18))
        input_layer = tf.keras.layers.concatenate([xy_input_layer, data_input_layer])
    
    x = tf.keras.layers.Dense(units, activation='tanh')(input_layer)
    
    for _ in range(Numlayers-1):
        x = tf.keras.layers.Dense(units, activation='tanh')(x)
    
    x = tf.keras.layers.Dense(out_dim, activation='linear')(x)
    
    if model_name == 'SNET':
        model = tf.keras.models.Model(input_layer, x, name=model_name)
    else:
        model = tf.keras.models.Model([xy_input_layer, data_input_layer], x, name=model_name)
    model.summary()
    return model

def res_mlp(units=50, Numlayers=3, actv_fn='tanh', out_dim=1, model_name='GATED_NET'):
    xy_input_layer = tf.keras.layers.Input(shape=(2))
    
    encoder_1 = tf.keras.layers.Dense(units, activation=actv_fn)(xy_input_layer)
    encoder_2 = tf.keras.layers.Dense(units, activation=actv_fn)(xy_input_layer)
    
    z = tf.keras.layers.Dense(units, activation=actv_fn)(xy_input_layer)
    x = tf.math.multiply(z, encoder_1) + tf.math.multiply(1 - z, encoder_2)
    
    for i in range(Numlayers-1):
        z = tf.keras.layers.Dense(units, activation=actv_fn)(x)
        x = tf.math.multiply(z, encoder_1) + tf.math.multiply(1 - z, encoder_2)
    
    x = tf.keras.layers.Dense(out_dim, activation='linear')(x)
    model = tf.keras.models.Model(xy_input_layer, x, name=model_name)
    return model

def tflog10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def eval_error(u, u_gt, Omega):
    err_data = tf.math.multiply((u - u_gt), Omega)
    err = tf.reduce_mean(tf.square(err_data))
    
    flat_truth = tf.reshape(u_gt, [-1])
    mx = tf.math.reduce_max(flat_truth)
    mx = tf.cond(mx == tf.constant(0.0, tf.float64),
                 true_fn=lambda: tf.math.reduce_max(tf.math.abs(flat_truth)),
                 false_fn=lambda: tf.math.reduce_max(flat_truth))
    
    psnr = 10 * tflog10(tf.math.pow(mx, 2)/err)
    return err, psnr, err_data

def get_mse(gt, pred, Omega):
    return tf.reduce_mean(tf.square(tf.math.multiply((gt - pred), Omega)))

#########################

def ReadData(fileName):
    Dict = scipy.io.loadmat(fileName)
    return Dict

def ExtractX(Dict):
    YOMEG = Dict['YOmeg']
    XOMEG = Dict['XOmeg']
    Xall = np.zeros((XOMEG.shape[0], 2), dtype=np.int64)
    Xall[:, 0] = np.transpose(XOMEG)
    Xall[:, 1] = np.transpose(YOMEG)
    XallRand = ShuffleX(Xall)
    return Xall, XallRand

def ExtractX2(Dict):
    YOMEG = Dict['Yall']
    XOMEG = Dict['Xall']
    Xall = np.zeros((XOMEG.shape[0], 2), dtype=np.int64)
    Xall[:, 0] = np.transpose(XOMEG)
    Xall[:, 1] = np.transpose(YOMEG)
    return Xall

def ShuffleX(Xt):
    Num = Xt.shape[0]
    list_ = np.linspace(0, Num-1, Num)
    np.random.shuffle(list_)
    XRand = np.zeros_like(Xt)
    for i in range(Num):
        idx = int(list_[i])
        XRand[i, 0] = Xt[idx,0]
        XRand[i, 1] = Xt[idx,1]
    return XRand

def SetX(Xall, CorrectB, CorrectS, inputSize=2):
    Xnor = np.zeros([Xall.shape[0], inputSize])
    Xnor[:,0] = (Xall[:,0]-CorrectB)/CorrectS
    Xnor[:,1] = (Xall[:,1]-CorrectB)/CorrectS
    return Xnor

def CalcValuesCompact(Xpool, Dict, keys):
    tempVar = np.zeros((Xpool.shape[0], 1), dtype=np.float64)
    tempMat = np.zeros_like(Dict['Sigma'], dtype=np.float64)
    aVec =[]
    AVec = []
    D = {}
    
    for i in range(len(keys)):
        a = copy.deepcopy(tempVar)
        aVec.append(a)
        curKey = keys[i]
        if curKey in Dict.keys():
            AVec.append(Dict[curKey])
        else:
            AVec.append(tempMat)
    
    for i in range(Xpool.shape[0]):
        y = Xpool[i, 1] - 1
        x = Xpool[i, 0] - 1
        
        for k in range(len(keys)):
            A = AVec[k]
            aVec[k][i] = A[y,x]
    
    for k in range(len(keys)):
        a = aVec[k]
        a = np.squeeze(a)
        a = np.reshape(a, [Xpool.shape[0], 1])
        D.update({keys[k]: a})
    
    return D

def get_2d(key, Dict, Dict_small):
    Xall = Dict['Xall']
    all_pts = np.zeros((len(Xall), 1))
    for i in range(Xall.shape[0]):
        x = Xall[i, 0] - 1
        y = Xall[i, 1] - 1
        all_pts[i, :] = Dict_small[key][y, x]
    return all_pts

def get_2d_dict(Dict):
    dict_2d = dict({})
    dict_2d.update({'Sigma': get_2d('Sigma', Dict[0], Dict[0]), 
                   'sx': get_2d('sx', Dict[0], Dict[0]), 
                   'sy': get_2d('sy', Dict[0], Dict[0])})

    for i in range(8):
        all_u = get_2d('U', Dict[0], Dict[i])
        all_ux = get_2d('ux', Dict[0], Dict[i])
        all_uy = get_2d('uy', Dict[0], Dict[i])
        all_uxx = get_2d('uxx', Dict[0], Dict[i])
        all_uyy = get_2d('uyy', Dict[0], Dict[i])
        dict_2d.update({'U'+str(i): all_u,
                        'ux'+str(i): all_ux,
                        'uy'+str(i): all_uy,
                        'uxx'+str(i): all_uxx,
                        'uyy'+str(i): all_uyy,
                       })
    
    return dict_2d

#########################

def np2tf(xin):
    return tf.convert_to_tensor(xin, dtype=tf.float64)

def find_min_dist(arr1, arr2):
    P = np.add.outer(np.sum(arr1**2, axis=1), np.sum(arr2**2, axis=1))
    N = np.dot(arr1, arr2.T)
    dists = np.sqrt(P - 2*N)
    min_id = np.argmin(dists, axis=0)
    return min_id


# In[ ]:


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_params(main_path, params):
    csvpath = main_path + 'hparams.csv'
    
    if os.path.exists(csvpath):
        with open(csvpath, 'a') as f:
            w = csv.DictWriter(f, params.keys())
            w.writerow(params)
    else:
        with open(csvpath, 'w') as f:
            w = csv.DictWriter(f, params.keys())
            w.writeheader()
            w.writerow(params)
    
    print('Parameters Saved', flush=True)


# In[ ]:


##############################################
    
# draw_figs(all_d[0]['XYd'], all_d[0]['ud'], 'Ud', 'tripcolor')

def draw_figs(xypts, data, plot_title='Plot', mode='tripcolor'):
    if mode == 'tripcolor':
        plt.figure(figsize=(7,6))
        im = plt.tripcolor(xypts[:, 0], xypts[:, 1], np.squeeze(data))
        plt.colorbar(im)
        plt.title(plot_title)
        plt.show()
    elif mode == 'scatter':
        plt.figure(figsize=(7,6))
        im = plt.scatter(xypts[:, 0], xypts[:, 1], np.squeeze(data))
        plt.colorbar(im)
        plt.title(plot_title)
        plt.show()
    else:
        plt.figure(figsize=(7,6))
        im = plt.imshow(data)
        plt.colorbar(im)
        plt.title(plot_title)
        plt.show()

def plot_data_u(u_list, u_err_list, err_list, Dict, losses, save_results_dir, epoch, expn):
    ucap, uxcap, uycap, uxxcap, uyycap = u_list[0], u_list[1], u_list[2], u_list[3], u_list[4]
    ucaperr, uxcaperr, uycaperr, uxxcaperr, uyycaperr = u_err_list[0], u_err_list[1], u_err_list[2], u_err_list[3], u_err_list[4]
    uerrval, uxerrval, uyerrval, uxxerrval, uyyerrval = err_list[0], err_list[1], err_list[2], err_list[3], err_list[4]
    pde_loss = err_list[5]
    
    plt.figure(figsize=(15,20))
    plt.subplot(5,3,1)
    plt.colorbar(plt.imshow(np.multiply(Dict[expn]['U'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('U Gt')
    plt.tight_layout()
    plt.subplot(5,3,2)
    plt.colorbar(plt.imshow(ucap, cmap='viridis'))
    plt.title('U Pred')
    plt.tight_layout()
    plt.subplot(5,3,3)
    plt.colorbar(plt.imshow(ucaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}, PDE: {:0.4f}'.format(uerrval, pde_loss))
    plt.tight_layout()
    
    plt.subplot(5,3,4)
    plt.colorbar(plt.imshow(np.multiply(Dict[expn]['ux'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('UX Gt')
    plt.tight_layout()
    plt.subplot(5,3,5)
    plt.colorbar(plt.imshow(uxcap, cmap='viridis'))
    plt.title('UX Pred')
    plt.tight_layout()
    plt.subplot(5,3,6)
    plt.colorbar(plt.imshow(uxcaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(uxerrval))
    plt.tight_layout()
    
    plt.subplot(5,3,7)
    plt.colorbar(plt.imshow(np.multiply(Dict[expn]['uy'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('UY Gt')
    plt.tight_layout()
    plt.subplot(5,3,8)
    plt.colorbar(plt.imshow(uycap, cmap='viridis'))
    plt.title('UY Pred')
    plt.tight_layout()
    plt.subplot(5,3,9)
    plt.colorbar(plt.imshow(uycaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(uyerrval))
    plt.tight_layout()
    
    plt.subplot(5,3,10)
    plt.colorbar(plt.imshow(np.multiply(Dict[expn]['uxx'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('UXX Gt')
    plt.tight_layout()
    plt.subplot(5,3,11)
    plt.colorbar(plt.imshow(uxxcap, cmap='viridis'))
    plt.title('UXX Pred')
    plt.tight_layout()
    plt.subplot(5,3,12)
    plt.colorbar(plt.imshow(uxxcaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(uxxerrval))
    plt.tight_layout()
    
    plt.subplot(5,3,13)
    plt.colorbar(plt.imshow(np.multiply(Dict[expn]['uyy'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('UYY Gt')
    plt.tight_layout()
    plt.subplot(5,3,14)
    plt.colorbar(plt.imshow(uyycap, cmap='viridis'))
    plt.title('UYY Pred')
    plt.tight_layout()
    plt.subplot(5,3,15)
    plt.colorbar(plt.imshow(uyycaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(uyyerrval))
    plt.tight_layout()
    
    plt.savefig(save_results_dir + 'u_' + str(epoch) + '.png', dpi=150)
    plt.close()
    
    plot_u_loss_logs(losses, epoch, save_results_dir)
    

def plot_data_s(s_list, s_err_list, err_list, Dict, losses, save_results_dir, epoch):
    scap, sxcap, sycap = s_list[0], s_list[1], s_list[2]
    scaperr, sxcaperr, sycaperr = s_err_list[0], s_err_list[1], s_err_list[2]
    serrval, sxerrval, syerrval = err_list[0], err_list[1], err_list[2]
    pde_loss = err_list[3]
    
    plt.figure(figsize=(14,12))
    plt.subplot(3,3,1)
    plt.colorbar(plt.imshow(np.multiply(Dict[0]['Sigma'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('S Gt')
    plt.tight_layout()
    plt.subplot(3,3,2)
    plt.colorbar(plt.imshow(scap, cmap='viridis'))
    plt.title('S Pred')
    plt.tight_layout()
    plt.subplot(3,3,3)
    plt.colorbar(plt.imshow(scaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}, PDE: {:0.4f}'.format(serrval, pde_loss))
    plt.tight_layout()
    
    plt.subplot(3,3,4)
    plt.colorbar(plt.imshow(np.multiply(Dict[0]['sx'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('SX Gt')
    plt.tight_layout()
    plt.subplot(3,3,5)
    plt.colorbar(plt.imshow(sxcap, cmap='viridis'))
    plt.title('SX Pred')
    plt.tight_layout()
    plt.subplot(3,3,6)
    plt.colorbar(plt.imshow(sxcaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(sxerrval))
    plt.tight_layout()
    
    plt.subplot(3,3,7)
    plt.colorbar(plt.imshow(np.multiply(Dict[0]['sy'], Dict[0]['Omega']), cmap='viridis'))
    plt.title('SY Gt')
    plt.tight_layout()
    plt.subplot(3,3,8)
    plt.colorbar(plt.imshow(sycap, cmap='viridis'))
    plt.title('SY Pred')
    plt.tight_layout()
    plt.subplot(3,3,9)
    plt.colorbar(plt.imshow(sycaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(syerrval))
    plt.tight_layout()
    
    plt.savefig(save_results_dir + 'sigma_' + str(epoch) + '.png', dpi=150)
    plt.close()
    
    plot_s_loss_logs(losses, epoch, save_results_dir)

def plot_s_loss_logs(losses, epoch, save_results_dir):
    
    if epoch > 0:
        plt.figure(figsize=(9,12))
        x_axis = np.arange(0, epoch)
        plt.subplot(10, 1, 1)
        plt.plot(x_axis, losses[:epoch, 0])
        plt.ylabel('S Total')
        plt.tight_layout()
        
        plt.subplot(10, 1, 2)
        plt.plot(x_axis, losses[:epoch, 1])
        plt.ylabel('S L2')
        plt.tight_layout()
        
        plt.subplot(10, 1, 3)
        plt.plot(x_axis, losses[:epoch, 2])
        plt.ylabel('S Inf')
        plt.tight_layout()
        
        plt.subplot(10, 1, 4)
        plt.plot(x_axis, losses[:epoch, 3])
        plt.ylabel('S SXYPen')
        plt.tight_layout()

        plt.subplot(10, 1, 5)
        plt.plot(x_axis, losses[:epoch, 4])
        plt.ylabel('S NB')
        plt.tight_layout()
        
        plt.subplot(10, 1, 6)
        plt.plot(x_axis, losses[:epoch, 5])
        plt.ylabel('S Back B')
        plt.tight_layout()
        
        plt.subplot(10, 1, 7)
        plt.plot(x_axis, losses[:epoch, 6])
        plt.ylabel('S NE')
        plt.tight_layout()
        
        plt.subplot(10, 1, 8)
        plt.plot(x_axis, losses[:epoch, 7])
        plt.ylabel('S Back E')
        plt.tight_layout()
        
        plt.subplot(10, 1, 9)
        plt.plot(x_axis, losses[:epoch, 8])
        plt.ylabel('S Stats')
        plt.tight_layout()
        
        plt.subplot(10, 1, 10)
        plt.plot(x_axis, losses[:epoch, 9])
        plt.ylabel('S NonZero')
        plt.tight_layout()
        
        plt.savefig(save_results_dir + 'sigma_loss_' + str(epoch) + '.png', dpi=150)
        plt.close()
        
def plot_u_loss_logs(losses, epoch, save_results_dir):
    
    if epoch > 0:
        plt.figure(figsize=(7,12))
        x_axis = np.arange(0, epoch)
        plt.subplot(611)
        plt.plot(x_axis, losses[:epoch, 0])
        plt.ylabel('U Total')
        plt.tight_layout()
        
        plt.subplot(612)
        plt.plot(x_axis, losses[:epoch, 1])
        plt.ylabel('U L2')
        plt.tight_layout()
        
        plt.subplot(613)
        plt.plot(x_axis, losses[:epoch, 2])
        plt.ylabel('U Inf')
        plt.tight_layout()
        
        plt.subplot(614)
        plt.plot(x_axis, losses[:epoch, 3])
        plt.ylabel('U NB')
        plt.tight_layout()

        plt.subplot(615)
        plt.plot(x_axis, losses[:epoch, 4])
        plt.ylabel('U NE')
        plt.tight_layout()

        plt.subplot(616)
        plt.plot(x_axis, losses[:epoch, 5])
        plt.ylabel('U Dir')
        plt.tight_layout()
        
        plt.savefig(save_results_dir + 'u_loss_' + str(epoch) + '.png', dpi=150)
        plt.close()
        
        
def plot_loss_logs(losses, epoch, save_results_dir):
    
    if epoch > 0:
        plt.figure(figsize=(9,18))
        x_axis = np.arange(0, epoch)
        
        plt.subplot(18, 1, 1)
        plt.plot(x_axis, losses[:epoch, 0])
        plt.ylabel('PDE Total')
        plt.tight_layout()
        
        plt.subplot(18, 1, 2)
        plt.plot(x_axis, losses[:epoch, 1])
        plt.ylabel('U Total')
        plt.tight_layout()
        
        plt.subplot(18, 1, 3)
        plt.plot(x_axis, losses[:epoch, 2])
        plt.ylabel('S Total')
        plt.tight_layout()
        
        plt.subplot(18, 1, 4)
        plt.plot(x_axis, losses[:epoch, 3])
        plt.ylabel('PDE L2')
        plt.tight_layout()
        
        plt.subplot(18, 1, 5)
        plt.plot(x_axis, losses[:epoch, 4])
        plt.ylabel('PDE Inf')
        plt.tight_layout()
        
        plt.subplot(18, 1, 6)
        plt.plot(x_axis, losses[:epoch, 5])
        plt.ylabel('NB')
        plt.tight_layout()
        
        plt.subplot(18, 1, 7)
        plt.plot(x_axis, losses[:epoch, 6])
        plt.ylabel('NE')
        plt.tight_layout()
        
        plt.subplot(18, 1, 8)
        plt.plot(x_axis, losses[:epoch, 7])
        plt.ylabel('Dirc B')
        plt.tight_layout()
        
        plt.subplot(18, 1, 9)
        plt.plot(x_axis, losses[:epoch, 8])
        plt.ylabel('Dirc E')
        plt.tight_layout()
        
        plt.subplot(18, 1, 10)
        plt.plot(x_axis, losses[:epoch, 9])
        plt.ylabel('S Param')
        plt.tight_layout()
        
        plt.subplot(18, 1, 11)
        plt.plot(x_axis, losses[:epoch, 10])
        plt.ylabel('U Param')
        plt.tight_layout()
        
        plt.subplot(18, 1, 12)
        plt.plot(x_axis, losses[:epoch, 11])
        plt.ylabel('SXY Pen')
        plt.tight_layout()
        
        plt.subplot(18, 1, 13)
        plt.plot(x_axis, losses[:epoch, 12])
        plt.ylabel('Total B')
        plt.tight_layout()
        
        plt.subplot(18, 1, 14)
        plt.plot(x_axis, losses[:epoch, 13])
        plt.ylabel('Total E')
        plt.tight_layout()
        
        plt.subplot(18, 1, 15)
        plt.plot(x_axis, losses[:epoch, 14])
        plt.ylabel('Prior')
        plt.tight_layout()
        
        plt.subplot(18, 1, 16)
        plt.plot(x_axis, losses[:epoch, 15])
        plt.ylabel('MaxSigma')
        plt.tight_layout()
        
        plt.subplot(18, 1, 17)
        plt.plot(x_axis, losses[:epoch, 16])
        plt.ylabel('Avg Match')
        plt.tight_layout()
        
        plt.subplot(18, 1, 18)
        plt.plot(x_axis, losses[:epoch, 17])
        plt.ylabel('Enforce 1')
        plt.tight_layout()
        
        plt.savefig(save_results_dir + 'loss_' + str(epoch) + '.png', dpi=150)
        plt.close()
