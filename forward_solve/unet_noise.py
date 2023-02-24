#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
from utils import np2tf, eval_error, make_dirs, SetX, CalcValuesCompact, res_mlp, get_mse

from keras import backend as K
K.clear_session()

import gc
gc.collect()


# In[ ]:


trial = 5

e_number = 0

save_results_dir = './noisy_fwd_lvl3/trial_' + str(trial) + '/'
make_dirs(save_results_dir)
print('\n', save_results_dir)

# ppath = '../dataset/multi_phantom_5_128.npy'
# ppath = '../dataset/multi_phantom_1_128.npy'
# ppath = '../dataset/multi_phantom_11_128.npy'
# ppath = '../dataset/multi_phantom_3_128.npy'
ppath = '../dataset/multi_phantom_4_128.npy'

all_d = np.load(ppath, allow_pickle=True).tolist()
h = all_d[0]['h'][0][0]

print(len(all_d))
print(all_d[0].keys())
CorrectS = all_d[0]['CorrectS']
CorrectB = all_d[0]['CorrectB']
XallRand = all_d[0]['XallRand']

print(ppath, flush=True)


# In[ ]:


sample_rate = 20
percent = round(len(all_d[0]['xyb']) * (sample_rate/100))
rand_indx = np.random.randint(low = 0, high = len(all_d[0]['xyb']) , size = percent)

noise_factor = 0.5 # change this as per paper
noisy_ue = all_d[e_number]['ue'] + noise_factor * np.random.uniform(all_d[e_number]['ue'].shape)

boundary_dict = dict({'xyb': np2tf(all_d[0]['xyb']), 
                      'sb': np2tf(all_d[0]['sb']),
                      'ub': np2tf(all_d[e_number]['ub']),
                      
                      'xyb_rand': np2tf(all_d[0]['xyb'][rand_indx]), 
                      'ub_rand': np2tf(all_d[e_number]['ub'][rand_indx]),
                      
                      'xye': np2tf(all_d[0]['xye']),
                      'ue': np2tf(noisy_ue),
                      'se': np2tf(all_d[0]['se']),
                      'fe': np2tf(all_d[e_number]['fe']),
                     })

BatchSize = 1000
Xpool = np.zeros((BatchSize, 2), dtype=np.int64)
MaxNumOfPoints = len(XallRand)
NumOfLoops = int(np.ceil(min(XallRand.shape[0], MaxNumOfPoints)/BatchSize))
print(NumOfLoops)

total_epochs = 3000
decay_every = 200
d_steps = NumOfLoops*decay_every
init_lr = 0.005
dec_r = 0.9
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, decay_steps=d_steps, decay_rate=dec_r, staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

print('\n', flush=True)
print('\n', flush=True)

print("\nbatch_size: {}, total_epochs: {}, decay_every: {}, d_steps: {}, init_lr: {}, dec_r: {}, opt: {}".format(BatchSize, 
                                                                                                                 total_epochs, decay_every, 
                                                                                                                 d_steps, init_lr, dec_r, 'Adam'), flush=True)

K = 40; pdel2 = 0.1; pdeinf = 0.05; nbl = 1; nel = 0.1; pl = 1e-06; dircl = 100; uxypen = 0; ppow = 1;

temp = '\nK: {}, pdeinf: {}, pdel2: {}, nbl: {}, nel: {}, pl: {}, dircl: {}, ppow: {}, uxypen: {}, e_number: {}, noise_factor: {}'
print(temp.format(K, pdeinf, pdel2, nbl, nel, pl, dircl, ppow, uxypen, e_number, noise_factor), flush=True)
print('\n', flush=True)
print('\n', flush=True)

unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
unet.summary()


# In[ ]:


@tf.function
def compute_2order(xy):
    ucap = unet(xy)
    uxy = tf.gradients(ucap, xy)[0]
    ux = tf.expand_dims(uxy[:, 0], 1)
    uy = tf.expand_dims(uxy[:, 1], 1)
    
    uxxyy = tf.gradients(ux, xy)[0]
    uxx = tf.expand_dims(uxxyy[:, 0], 1)
    
    uxxyy = tf.gradients(uy, xy)[0]
    uyy = tf.expand_dims(uxxyy[:, 1], 1)
    return ucap, ux, uy, uxx, uyy

@tf.function
def compute_1order(xy):
    ucap = unet(xy)
    uxy = tf.gradients(ucap, xy)[0]
    
    ux = tf.expand_dims(uxy[:, 0], 1)
    uy = tf.expand_dims(uxy[:, 1], 1)
    return ux, uy

def pde_loss(xy, s, sx, sy, K, pdeinf, pdel2):
    # - div . (sigma delta u)
    _, ux, uy, uxx, uyy = compute_2order(xy)
    
    pred_pde = sx * ux/h + sy * uy/h + s * (uxx + uyy)
    # print(pred_pde.shape, sx.shape, ux.shape, sy.shape, uy.shape, s.shape, uxx.shape, uyy.shape)
    
    topk = tf.nn.top_k(tf.reshape(tf.abs(pred_pde), (-1,)), K)
    loss_inf = pdeinf * tf.reduce_mean(topk.values)
    loss_l2 = pdel2 * tf.reduce_mean(tf.square(pred_pde))
    
    loss_uxpen = uxypen * tf.reduce_mean(tf.math.pow(tf.square(ux/h), ppow))
    loss_uypen = uxypen * tf.reduce_mean(tf.math.pow(tf.square(uy/h), ppow))
    loss_uxxpen = uxypen * tf.reduce_mean(tf.math.pow(tf.square(uxx), ppow))
    loss_uyypen = uxypen * tf.reduce_mean(tf.math.pow(tf.square(uyy), ppow))
    losslp = loss_uxpen + loss_uypen + loss_uxxpen + loss_uyypen
    return loss_inf, loss_l2, losslp

def nuemann_loss(xy, sb, nbl):
    # conduc * udn
    ux, uy = compute_1order(xy)
    
    # nuemann loss boundary
    pred_nue = sb * (ux + uy)
    # print(pred_nue.shape, ux.shape, uy.shape, sb.shape, nhatx.shape, nhaty.shape)
    
    return nbl * tf.reduce_mean(tf.abs(pred_nue))

def nuem_loss_electrode(xy, current_e, se, nel):
    # conduc * udn - f
    ux, uy = compute_1order(xy)
    
    # at electrodes only
    pred_nue = se * (ux + uy) - current_e
    # print(pred_nue.shape, ux.shape, uy.shape, se.shape, nhatx.shape, nhaty.shape, current_e.shape)
    
    return nel * tf.reduce_mean(tf.abs(pred_nue))

def compute_param_loss(model, pl):
    return pl * tf.math.add_n([tf.nn.l2_loss(v) for v in model.weights if 'bias' not in v.name])

def dirch_loss(xy, ge, dirc):
    # u - g
    ucap = unet(xy)
    
    # dirichlet loss electrode
    pred_dirch = ucap - ge
    # print(pred_dirch.shape, ucap.shape, ge.shape)
    
    return dircl * tf.reduce_mean(tf.abs(pred_dirch))


# In[ ]:


@tf.function
def train_step(xy, dd, b_dict, K, pdeinf, pdel2, nbl, nel, pl, dircl):
    
    with tf.GradientTape() as tape:
        
        s, sx, sy = np2tf(dd['Sigma']), np2tf(dd['sx']), np2tf(dd['sy'])
        
        loss_inf, loss_l2, losslp = pde_loss(xy, s, sx, sy, K, pdeinf, pdel2)
        
        loss_dir_e = dirch_loss(b_dict['xye'], b_dict['ue'], dircl)
        
        loss_dir_b = dirch_loss(b_dict['xyb_rand'], b_dict['ub_rand'], dircl)
        
        loss_nb = nuemann_loss(b_dict['xyb'], b_dict['sb'], nbl)
        
        loss_ne = nuem_loss_electrode(b_dict['xye'], b_dict['fe'], b_dict['se'], nel)
        
        param_loss = compute_param_loss(unet, pl)
        
        total_loss = loss_l2 + loss_inf + param_loss + loss_nb + loss_ne + loss_dir_e + losslp
    
    grads = tape.gradient(total_loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(grads, unet.trainable_variables))
    return total_loss, loss_l2, loss_inf, loss_dir_e, loss_dir_b, loss_nb, loss_ne, param_loss


# In[ ]:


def get_data():
    Xallnor = all_d[0]['XYallnor']
    ucap, ux, uy, uxx, uyy = compute_2order(Xallnor)
    
    ucap = ucap.numpy();
    ux = ux.numpy()*h; uy = uy.numpy()*h; 
    uxx = uxx.numpy()*h*h; uyy = uyy.numpy()*h*h;
    
    ucap = ucap.reshape(128,128, order='F') * all_d[0]['Omega']
    ux = ux.reshape(128,128, order='F') * all_d[0]['Omega']
    uy = uy.reshape(128,128, order='F') * all_d[0]['Omega']
    uxx = uxx.reshape(128,128, order='F') * all_d[0]['Omega']
    uyy = uyy.reshape(128,128, order='F') * all_d[0]['Omega']
    return ucap, ux, uy, uxx, uyy

def get_u():
    ucap = unet(all_d[0]['XYallnor'])
    ucap = tf.transpose(tf.reshape(ucap, [128,128]))
    mse = get_mse(ucap, all_d[e_number]['U'], all_d[0]['Omega'])
    return mse

def plot_data(u_list, u_err_list, err_list, Dict, losses, save_results_dir, epoch):
    ucap, uxcap, uycap, uxxcap, uyycap = u_list[0], u_list[1], u_list[2], u_list[3], u_list[4]
    ucaperr, uxcaperr, uycaperr, uxxcaperr, uyycaperr = u_err_list[0], u_err_list[1], u_err_list[2], u_err_list[3], u_err_list[4]
    uerrval, uxerrval, uyerrval, uxxerrval, uyyerrval = err_list[0], err_list[1], err_list[2], err_list[3], err_list[4]
    
    plt.figure(figsize=(15,20))
    plt.subplot(5,3,1)
    plt.colorbar(plt.imshow(Dict[e_number]['U'], cmap='viridis'))
    plt.title('U Gt')
    plt.tight_layout()
    plt.subplot(5,3,2)
    plt.colorbar(plt.imshow(ucap, cmap='viridis'))
    plt.title('U Pred')
    plt.tight_layout()
    plt.subplot(5,3,3)
    plt.colorbar(plt.imshow(ucaperr, cmap='viridis'))
    plt.title('MSE: {:0.4f}'.format(uerrval))
    plt.tight_layout()
    
    plt.subplot(5,3,4)
    plt.colorbar(plt.imshow(Dict[e_number]['ux'], cmap='viridis'))
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
    plt.colorbar(plt.imshow(Dict[e_number]['uy'], cmap='viridis'))
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
    plt.colorbar(plt.imshow(Dict[e_number]['uxx'], cmap='viridis'))
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
    plt.colorbar(plt.imshow(Dict[e_number]['uyy'], cmap='viridis'))
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
    
    if epoch > 0:
        plt.figure(figsize=(8,12))
        x_axis = np.arange(0, epoch)
        plt.subplot(811)
        plt.plot(x_axis, losses[:epoch, 0])
        plt.ylabel('Total')
        plt.tight_layout()
        
        plt.subplot(812)
        plt.plot(x_axis, losses[:epoch, 1])
        plt.ylabel('L2')
        plt.tight_layout()
        
        plt.subplot(813)
        plt.plot(x_axis, losses[:epoch, 2])
        plt.ylabel('Inf')
        plt.tight_layout()
        
        plt.subplot(814)
        plt.plot(x_axis, losses[:epoch, 3])
        plt.ylabel('Dir E')
        plt.tight_layout()
        
        plt.subplot(815)
        plt.plot(x_axis, losses[:epoch, 3])
        plt.ylabel('Dir B')
        plt.tight_layout()
        
        plt.subplot(816)
        plt.plot(x_axis, losses[:epoch, 4])
        plt.ylabel('Nue B')
        plt.tight_layout()

        plt.subplot(817)
        plt.plot(x_axis, losses[:epoch, 5])
        plt.ylabel('Nue E')
        plt.tight_layout()
        
        plt.subplot(818)
        plt.plot(x_axis, losses[:epoch, 6])
        plt.ylabel('Param')
        plt.tight_layout()
        
        plt.savefig(save_results_dir + 'loss_' + str(epoch) + '.png', dpi=150)
        plt.close()


# In[ ]:


best_loss = 1e20
best_epoch = 0
loss_log = np.zeros((total_epochs, 8))

for epoch in range(total_epochs):
    
    mean_val_coarse = tf.reduce_mean(unet(all_d[0]['xyd']))
    
    total_loss = 0;    total_l2 = 0;    total_inf = 0;    total_dir = 0;    total_nb = 0;    total_ne = 0;    total_param = 0;    total_dirb = 0;
    for j in range(NumOfLoops):
        j1 = j * BatchSize
        j2 = min(XallRand.shape[0], (j + 1) * BatchSize)
        
        Xpool[0:(j2 - j1), 0] = np.transpose(XallRand[j1:j2, 0])
        Xpool[0:(j2 - j1), 1] = np.transpose(XallRand[j1:j2, 1])
        
        Xnor_ = SetX(Xpool, CorrectB, CorrectS)
        
        D = CalcValuesCompact(Xpool, all_d[0], ['Sigma', 'sx', 'sy'])
        
        total_l, pde_l, loss_inf, loss_dir_e, loss_dir_b, loss_nb, loss_ne, param_loss = train_step(Xnor_, D, boundary_dict, K, 
                                                                                                    pdeinf, pdel2, nbl, nel, pl, dircl)
        
        total_loss += total_l;        total_l2 += pde_l;               total_inf += loss_inf;         total_nb += loss_nb;          
        total_dir += loss_dir_e;      total_dirb += loss_dir_b;        total_param += param_loss;       total_ne += loss_ne;
    
    loss_log[epoch] = np.array([total_loss.numpy(), total_l2.numpy(), total_inf.numpy(), total_dir.numpy(), 
                                total_dirb.numpy(), total_nb.numpy(), total_ne.numpy(), total_param.numpy()])
    
    err = get_u()
    
    if total_loss < best_loss:
        best_loss = total_loss
        best_epoch = epoch
        unet.save_weights(save_results_dir + 'unet_best.h5')
    
    if epoch % 200 == 0:
        curr_lr = optimizer.learning_rate(optimizer.iterations)
        print("\nEpoch: {}, Total Loss: {}, LR: {}".format(epoch, total_loss.numpy(), curr_lr), flush=True)
        print("Best Epoch: {}, Best Loss: {}".format(best_epoch, best_loss), flush=True)
        print("PDE Loss: {}, Inf: {}".format(total_l2.numpy(), total_inf.numpy()), flush=True)
        print("Dir E: {}, Dir B: {}".format(total_dir.numpy(), total_dirb.numpy()), flush=True)
        print("NE: {}, NB: {}, Param: {}".format(total_ne.numpy(), total_nb.numpy(), total_param.numpy()), flush=True)
        
        ucap, uxcap, uycap, uxxcap, uyycap = get_data()
        err, psnr, uerr = eval_error(ucap, all_d[e_number]['U'], all_d[0]['Omega'])
        errx, psnrx, uxerr = eval_error(uxcap, all_d[e_number]['ux'], all_d[0]['Omega'])
        erry, psnry, uyerr = eval_error(uycap, all_d[e_number]['uy'], all_d[0]['Omega'])
        errxx, psnrxx, uxxerr = eval_error(uxxcap, all_d[e_number]['uxx'], all_d[0]['Omega'])
        erryy, psnryy, uyyerr = eval_error(uyycap, all_d[e_number]['uyy'], all_d[0]['Omega'])
        
        print("MSE u: {}, PSNR u: {}".format(err, psnr), flush=True)
        print("MSE ux: {}, PSNR ux: {}".format(errx, psnrx), flush=True)
        print("MSE uy: {}, PSNR uy: {}".format(erry, psnry), flush=True)
        print("MSE uxx: {}, PSNR uxx: {}".format(errxx, psnrxx), flush=True)
        print("MSE uyy: {}, PSNR uyy: {}".format(erryy, psnryy), flush=True)
        
        ulist = [ucap, uxcap, uycap, uxxcap, uyycap]
        uerr_list = [uerr, uxerr, uyerr, uxxerr, uyyerr]
        err_list = [err, errx, erry, errxx, erryy]
        
        plot_data(ulist, uerr_list, err_list, all_d, loss_log, save_results_dir, epoch)


# In[ ]:

err = get_u()

if total_loss < best_loss:
    best_loss = total_loss
    best_epoch = epoch
    unet.save_weights(save_results_dir + 'unet_best.h5')

print("\nEpoch: {}, Total Loss: {}, LR: {}".format(epoch, total_loss.numpy(), curr_lr), flush=True)
print("Best Epoch: {}, Best Loss: {}".format(best_epoch, best_loss), flush=True)
print("PDE Loss: {}, Inf: {}".format(total_l2.numpy(), total_inf.numpy()), flush=True)
print("Dir E: {}, Dir B: {}".format(total_dir.numpy(), total_dirb.numpy()), flush=True)
print("NE: {}, NB: {}, Param: {}".format(total_ne.numpy(), total_nb.numpy(), total_param.numpy()), flush=True)


# In[ ]:


unet.load_weights(save_results_dir + 'unet_best.h5')

ucap, uxcap, uycap, uxxcap, uyycap = get_data()
err, psnr, uerr = eval_error(ucap, all_d[e_number]['U'], all_d[0]['Omega'])
errx, psnrx, uxerr = eval_error(uxcap, all_d[e_number]['ux'], all_d[0]['Omega'])
erry, psnry, uyerr = eval_error(uycap, all_d[e_number]['uy'], all_d[0]['Omega'])
errxx, psnrxx, uxxerr = eval_error(uxxcap, all_d[e_number]['uxx'], all_d[0]['Omega'])
erryy, psnryy, uyyerr = eval_error(uyycap, all_d[e_number]['uyy'], all_d[0]['Omega'])

print("MSE u: {}, PSNR u: {}".format(err, psnr), flush=True)
print("MSE ux: {}, PSNR ux: {}".format(errx, psnrx), flush=True)
print("MSE uy: {}, PSNR uy: {}".format(erry, psnry), flush=True)
print("MSE uxx: {}, PSNR uxx: {}".format(errxx, psnrxx), flush=True)
print("MSE uyy: {}, PSNR uyy: {}".format(erryy, psnryy), flush=True)

ulist = [ucap, uxcap, uycap, uxxcap, uyycap]
uerr_list = [uerr, uxerr, uyerr, uxxerr, uyyerr]
err_list = [err, errx, erry, errxx, erryy]

plot_data(ulist, uerr_list, err_list, all_d, loss_log, save_results_dir, epoch)

np.save(save_results_dir + 'loss_log_u.npy', loss_log)


# In[ ]: