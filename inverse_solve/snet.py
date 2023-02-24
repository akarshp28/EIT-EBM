#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float64')
from utils import np2tf, eval_error, make_dirs, unsup_mlp, res_mlp, SetX, CalcValuesCompact, get_mse
from utils_prior import create_res_ebm_sigma

from keras import backend as K
K.clear_session()

import gc
gc.collect()


# In[ ]:


trial = 1
phantom = trial

run_exp = 'proposed'
prior = True
main_path = './energy_plot'

if run_exp == 'proposed':
    if prior:
        save_results_dir = main_path + '/proposed_prior/trial_' + str(trial) + '/'
    else:
        save_results_dir = main_path + '/proposed/trial_' + str(trial) + '/'
elif run_exp == 'dgm':
    if prior:
        save_results_dir = main_path + '/dgm_prior/trial_' + str(trial) + '/'
    else:
        save_results_dir = main_path + '/dgm/trial_' + str(trial) + '/'
elif run_exp == 'unsup':
    if prior:
        save_results_dir = main_path + '/unsup_prior/trial_' + str(trial) + '/'
    else:
        save_results_dir = main_path + '/unsup/trial_' + str(trial) + '/'
else:
    print('bruh', flush=True)

print('\n', flush=True)
print('\n', flush=True)
    
make_dirs(save_results_dir)
print('\n', save_results_dir)

print('prior', prior, flush=True)

e_number = 0

if phantom == 1:
    ppath = '../dataset/multi_phantom_5_128.npy'
    
    unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
    utrial = 1
    unetpath = './attemp1/fwd_paper_exp' + str(e_number) + '/trial_' + str(utrial) + '/unet_best.h5'
    print(unetpath, flush=True)
    unet.load_weights(unetpath)
    unet.trainable = False

elif phantom == 2:
    ppath = '../dataset/multi_phantom_1_128.npy'
    
    unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
    utrial = 2
    unetpath = './attemp1/fwd_paper_exp' + str(e_number) + '/trial_' + str(utrial) + '/unet_best.h5'
    print(unetpath, flush=True)
    unet.load_weights(unetpath)
    unet.trainable = False

elif phantom ==  3:
    ppath = '../dataset/multi_phantom_11_128.npy'
    
    unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
    utrial = 3
    unetpath = './attemp1/fwd_paper_exp' + str(e_number) + '/trial_' + str(utrial) + '/unet_best.h5'
    print(unetpath, flush=True)
    unet.load_weights(unetpath)
    unet.trainable = False
    
elif phantom ==  4:
    ppath = '../dataset/multi_phantom_3_128.npy'
    
    unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
    utrial = 4
    unetpath = './attemp1/fwd_paper_exp' + str(e_number) + '/trial_' + str(utrial) + '/unet_best.h5'
    print(unetpath, flush=True)
    unet.load_weights(unetpath)
    unet.trainable = False

elif phantom == 5:
    ppath = '../dataset/multi_phantom_4_128.npy'
    
    unet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_UNET')
    utrial = 5
    unetpath = './attemp1/fwd_paper_exp' + str(e_number) + '/trial_' + str(utrial) + '/unet_best.h5'
    print(unetpath, flush=True)
    unet.load_weights(unetpath)
    unet.trainable = False
    
else:
    print('Model and Data not selected', flush=True)

all_d = np.load(ppath, allow_pickle=True).tolist()
h = all_d[0]['h'][0][0]
CorrectS = all_d[0]['CorrectS']
CorrectB = all_d[0]['CorrectB']
XallRand = all_d[0]['XallRand']
coarse_sigma_mean = all_d[0]['sd'].mean()
print(coarse_sigma_mean, flush=True)
print(ppath, flush=True)


# In[ ]:


BatchSize = 1000
Xpool = np.zeros((BatchSize, 2), dtype=np.int64)
MaxNumOfPoints = len(XallRand)
NumOfLoops = int(np.ceil(min(XallRand.shape[0], MaxNumOfPoints)/BatchSize))

if run_exp == 'proposed':
    # Our proposed settings
    K = 40; pdel2 = 0.05; pdeinf = 0.05; nbl = 1; nel = 0.1; pl = 1e-06; en_fact = 0.0001; no_neg = 10; sxypen = 0.01; ppow = 1;
    dec_r = 0.9
elif run_exp == 'dgm':
    # DGM
    K = 40; pdel2 = 1; pdeinf = 0; nbl = 1; nel = 1; pl = 0; en_fact = 0.0001; no_neg = 10; sxypen = 0; ppow = 1;
    dec_r = 0.8
elif run_exp == 'unsup':
    # Unsup EIT
    K = 40; pdel2 = 0.01; pdeinf = 0.01; nbl = 1; nel = 1; pl = 1e-08; en_fact = 0.0001; no_neg = 10; sxypen = 0.01; ppow = 1;
    dec_r = 0.8
else:
    print('Select correct settings', flush=True)

total_epochs = 3000
decay_every = 200
d_steps = NumOfLoops*decay_every
init_lr = 0.005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(init_lr, decay_steps=d_steps, decay_rate=dec_r, staircase=True)
optimizer = tf.keras.optimizers.Adam(init_lr)

print("\nbatch_size: {}, total_epochs: {}, decay_every: {}, d_steps: {}, init_lr: {}, dec_r: {}, opt: {}".format(BatchSize, 
                                                                                                                 total_epochs, decay_every, 
                                                                                                                 d_steps, init_lr, 
                                                                                                                 dec_r, 'Adam'), flush=True)

print('\nK: {}, pdeinf: {}, pdel2: {}, nbl: {}, nel: {}, pl: {}, en_fact: {}, no_neg: {}, sxypen: {}, ppow: {}'.format(K, pdeinf, pdel2, nbl, nel, 
                                                                                                                       pl, en_fact, 
                                                                                                                       no_neg, sxypen, ppow), flush=True)
print('\n', flush=True)
print('\n', flush=True)

if run_exp == 'proposed':
    snet = res_mlp(units=64, Numlayers=4, actv_fn='tanh', out_dim=1, model_name='GATED_SNET')
else:
    snet = unsup_mlp(model_name='UNSUP_SNET')
snet.summary()


# In[ ]:


def get_upred_data():
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
def compute_1order(xy, uins=None):
    if uins == None:
        datacap = snet(xy)
    else:
        datacap = unet(xy)
    
    dxy = tf.gradients(datacap, xy)[0]
    dx = tf.expand_dims(dxy[:, 0], 1)
    dy = tf.expand_dims(dxy[:, 1], 1)
    return datacap, dx, dy


ucap, uxcap, uycap, uxxcap, uyycap = get_upred_data()

_, uxb, uyb = compute_1order(all_d[0]['xyb'], 0)
_, uxe, uye = compute_1order(all_d[0]['xye'], 0)

domain_dict = dict({'U'+str(e_number): ucap,
                    'ux'+str(e_number): uxcap,
                    'uy'+str(e_number): uycap,
                    'uxx'+str(e_number): uxxcap,
                    'uyy'+str(e_number): uyycap,
                    
                    'Xall': all_d[0]['Xall'],
                    'Xdall': all_d[0]['Xdall'],
                    'XallRand': all_d[0]['XallRand'],
                    'CorrectB': all_d[0]['CorrectB'],
                    'CorrectS': all_d[0]['CorrectS'],
                    'Sigma': all_d[0]['Sigma'],
                 })

boundary_dict = dict({'xyb': np2tf(all_d[0]['xyb']), 
                      'xye': np2tf(all_d[0]['xye']),
                      'uxe'+str(e_number): uxe,
                      'uye'+str(e_number): uye,
                      'fe'+str(e_number): np2tf(all_d[e_number]['fe']),
                      'uxb'+str(e_number): uxb,
                      'uyb'+str(e_number): uyb,
                     })

ul = ['U'+str(e_number), 'ux'+str(e_number), 'uy'+str(e_number), 'uxx'+str(e_number), 'uyy'+str(e_number)]

plt.imshow(uxcap)
plt.savefig(save_results_dir + 'upred.png')
plt.close()


# In[ ]:


@tf.function
def pde_loss(xy, ux, uy, uxx, uyy, K, pdeinf, pdel2):
    # - div . (sigma delta u)
    s, sx, sy = compute_1order(xy)
    
    # data was prepared with (*h)
    pred_pde = sx * ux/h + sy * uy/h + s * (uxx/(h*h) + uyy/(h*h))
    
    topk = tf.nn.top_k(tf.reshape(tf.abs(pred_pde), (-1,)), K)
    loss_inf = pdeinf * tf.reduce_mean(topk.values)
    loss_l2 = pdel2 * tf.reduce_mean(tf.square(pred_pde))
    return loss_inf, loss_l2, sx, sy

@tf.function
def nuemann_loss(xy, ux, uy, nbl):
    # conduc * udn
    sb = snet(xy)
    
    # nuemann loss boundary
    pred_nue = sb * (ux + uy)
    
    return nbl * tf.reduce_mean(tf.abs(pred_nue))

@tf.function
def nuem_loss_electrode(xy, ux, uy, current_e, nel):
    # conduc * udn - f
    se = snet(xy)
    
    # at electrodes only
    pred_nue = se * (ux + uy) - current_e
    
    return nel * tf.reduce_mean(tf.abs(pred_nue))

@tf.function
def compute_param_loss(model, pl):
    return pl * tf.math.add_n([tf.nn.l2_loss(v) for v in model.weights if 'bias' not in v.name])

@tf.function
def compute_loss_prior(en_fact):
    x = all_d[0]['XYallnor']
    scap = snet(x)
    scap = tf.transpose(tf.reshape(scap, [128,128])) * all_d[0]['Omega']
    scap = tf.reshape(scap, [1, -1])
    en = -ebm([scap, labels])
    g = tf.gradients(en, scap)[0]
    gn = tf.reduce_sum(tf.square(g))
    return en_fact * en, g, gn


# In[ ]:


ebm = create_res_ebm_sigma()
ebm.load_weights('/users/apokkunu/eit/eit_unsup/ebm_prior/score_matching/dsm/trial_1/ebm_best_600.h5')

labels = tf.ones([1, 1]) * 0.01
en_gt = -ebm([(all_d[0]['Sigma'] * all_d[0]['Omega']).reshape(1, -1), labels])
print(en_gt, flush=True)


# In[ ]:


@tf.function
def train_step(xy, dd, b_dict, K, pdeinf, pdel2, nbl, nel, pl, en_fact, no_neg):
    
    with tf.GradientTape() as tape:
        
        ux0, uy0 = np2tf(dd['ux' + str(e_number)]), np2tf(dd['uy' + str(e_number)])
        uxx0, uyy0 = np2tf(dd['uxx' + str(e_number)]), np2tf(dd['uyy' + str(e_number)])
        
        loss_inf0, loss_l20, sx_cap, sy_cap = pde_loss(xy, ux0, uy0, uxx0, uyy0, K, pdeinf, pdel2)
        
        loss_nb = nuemann_loss(b_dict['xyb'], 
                               b_dict['uxb' + str(e_number)], b_dict['uyb' + str(e_number)], nbl)
        
        loss_ne = nuem_loss_electrode(b_dict['xye'], 
                                      b_dict['uxe' + str(e_number)], b_dict['uye' + str(e_number)], 
                                      b_dict['fe' + str(e_number)], nel)
        
        param_loss = compute_param_loss(snet, pl)
        
        loss_sxpen = sxypen * tf.reduce_mean(tf.math.pow(tf.abs(sx_cap), ppow))
        loss_sypen = sxypen * tf.reduce_mean(tf.math.pow(tf.abs(sy_cap), ppow))
        losslp = loss_sxpen + loss_sypen
        
        loss_b = tf.reduce_mean(tf.abs(snet(b_dict['xyb']) - 1.))
        loss_e = tf.reduce_mean(tf.abs(snet(b_dict['xye']) - 1.))
        
        loss_prior, g, gn = compute_loss_prior(en_fact)
        
        loss_noneg = no_neg * tf.reduce_mean(tf.nn.relu(1 - snet(all_d[0]['Xallnor'])))
        
        total_loss = loss_inf0 + loss_l20 + param_loss + loss_b + loss_e + losslp + loss_nb + loss_ne + loss_noneg
        
        if prior:
            total_loss = total_loss + loss_prior
    
    grads = tape.gradient(total_loss, snet.trainable_variables)
    optimizer.apply_gradients(zip(grads, snet.trainable_variables))
    return total_loss, loss_l20, loss_inf0, losslp, loss_b, loss_e, loss_nb, loss_ne, loss_prior, loss_noneg, gn


# In[ ]:


def get_data():
    scap, sx, sy = compute_1order(all_d[0]['XYallnor'])
    scap = scap.numpy(); sx = sx.numpy()*h; sy = sy.numpy()*h;
    
    scap = scap.reshape(128,128, order='F') * all_d[0]['Omega']
    sx = sx.reshape(128,128, order='F') * all_d[0]['Omega']
    sy = sy.reshape(128,128, order='F') * all_d[0]['Omega']
    return scap, sx, sy

def get_s():
    scap = snet(all_d[0]['XYallnor']).numpy()
    scap = tf.transpose(tf.reshape(scap, [128,128]))
    mse = get_mse(scap, all_d[0]['Sigma'], all_d[0]['Omega'])
    return mse

def plot_data_s(s_list, s_err_list, err_list, Dict, save_results_dir, epoch):
    scap, sxcap, sycap = s_list[0], s_list[1], s_list[2]
    scaperr, sxcaperr, sycaperr = s_err_list[0], s_err_list[1], s_err_list[2]
    serrval, sxerrval, syerrval = err_list[0], err_list[1], err_list[2]
    
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
    plt.title('MSE Error: error = {:0.4f}'.format(serrval))
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
    plt.title('MSE Error: error = {:0.4f}'.format(sxerrval))
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
    plt.title('MSE Error: error = {:0.4f}'.format(syerrval))
    plt.tight_layout()
    
    plt.savefig(save_results_dir + 's_epoch_' + str(epoch) + '.png')
    plt.close()


# In[ ]:


best_loss = 1e20
best_epoch = 0
all_mse = []
s_loss_log = np.zeros((total_epochs, 11))

for epoch in range(total_epochs):
    
    mean_val_coarse = tf.reduce_mean(snet(all_d[0]['xyd']))
    
    total_loss = 0;    total_l2 = 0;    total_inf = 0;    total_lp = 0
    total_b = 0;    total_e = 0;    total_nb = 0;    total_ne = 0;     total_noneg = 0
    for j in range(NumOfLoops):
        j1 = j * BatchSize
        j2 = min(XallRand.shape[0], (j + 1) * BatchSize)
        
        Xpool[0:(j2 - j1), 0] = np.transpose(XallRand[j1:j2, 0])
        Xpool[0:(j2 - j1), 1] = np.transpose(XallRand[j1:j2, 1])
        
        Xnor_ = SetX(Xpool, CorrectB, CorrectS)
        
        D = CalcValuesCompact(Xpool, domain_dict, ul)
        
        loss, loss_l2, loss_inf, losslp, loss_b, loss_e, loss_nb, loss_ne, loss_prior, loss_noneg, gn = train_step(Xnor_, D, boundary_dict, 
                                                                                                               K, pdeinf, pdel2, nbl, nel, pl, 
                                                                                                               en_fact, no_neg)
        
        total_loss += loss;           total_l2 += loss_l2;        total_inf += loss_inf;        total_lp += losslp;         total_noneg += loss_noneg
        total_b += loss_b;            total_e += loss_e;          total_nb += loss_nb;          total_ne += loss_ne;
        
        
    
    err = get_s()
    all_mse.append(err)
    s_loss_log[epoch] = np.array([total_loss.numpy(), total_l2.numpy(), total_inf.numpy(), 
                                      total_lp.numpy(), total_b.numpy(), total_e.numpy(),
                                      total_nb.numpy(), total_ne.numpy(), loss_prior.numpy(),
                                      total_noneg.numpy(), gn.numpy()])
    
    if err < best_loss:
        best_loss = err
        best_epoch = epoch
        snet.save_weights(save_results_dir + 'snet_best.h5')
    
    if epoch % 200 == 0:
        curr_lr = optimizer.lr.numpy() #learning_rate(optimizer.iterations)
        print("\nEpoch: {}, Total Loss: {}, PDE Loss: {}, LR: {}".format(epoch, total_loss.numpy(), total_l2.numpy(), curr_lr), flush=True)
        print("Best Epoch: {}, Best MSE: {}".format(best_epoch, best_loss), flush=True)
        
        print("Inf: {}, SXPen: {}".format(total_inf.numpy(), total_lp.numpy()), flush=True)
        print("SEPen: {}, NE: {}".format(total_e.numpy(), total_ne.numpy()), flush=True)
        print("SBPen: {}, NB: {}, NoNeg: {}".format(total_b.numpy(), total_nb.numpy(), total_noneg.numpy()), flush=True)
        print("GT Avg: {}, Pred Avg: {}".format(coarse_sigma_mean, mean_val_coarse), flush=True)
        print("Gt En: {}, Pred En: {}, Pred En Grad: {}".format(en_gt.numpy(), loss_prior.numpy()/en_fact, gn.numpy()), flush=True)
        
        scap, sxcap, sycap = get_data()
        err, psnr, serr = eval_error(scap, all_d[0]['Sigma'], all_d[0]['Omega'])
        errx, psnrx, sxerr = eval_error(sxcap, all_d[0]['sx'], all_d[0]['Omega'])
        erry, psnry, syerr = eval_error(sycap, all_d[0]['sy'], all_d[0]['Omega'])
        
        slist = [scap, sxcap, sycap]
        serr_list = [serr, sxerr, syerr]
        err_list = [err, errx, erry]
        
        print("MSE s: {}, PSNR s: {}".format(err, psnr), flush=True)
        print("MSE sx: {}, PSNR sx: {}".format(errx, psnrx), flush=True)
        print("MSE sy: {}, PSNR sy: {}".format(erry, psnry), flush=True)
        
        plot_data_s(slist, serr_list, err_list, all_d, save_results_dir, epoch)


# In[ ]:


if err < best_loss:
    best_loss = err
    best_epoch = epoch
    snet.save_weights(save_results_dir + 'snet_best.h5')

scap, sxcap, sycap = get_data()
err, psnr, serr = eval_error(scap, all_d[0]['Sigma'], all_d[0]['Omega'])
errx, psnrx, sxerr = eval_error(sxcap, all_d[0]['sx'], all_d[0]['Omega'])
erry, psnry, syerr = eval_error(sycap, all_d[0]['sy'], all_d[0]['Omega'])

curr_lr = optimizer.lr.numpy()
print("\nEpoch: {}, Total Loss: {}, PDE Loss: {}, LR: {}".format(epoch, total_loss.numpy(), total_l2.numpy(), curr_lr), flush=True)
print("Best Epoch: {}, Best MSE: {}".format(best_epoch, best_loss), flush=True)
print("Inf: {}, SXPen: {}".format(total_inf.numpy(), total_lp.numpy()), flush=True)
print("SEPen: {}, NE: {}".format(total_e.numpy(), total_ne.numpy()), flush=True)
print("SBPen: {}, NB: {}".format(total_b.numpy(), total_nb.numpy()), flush=True)
print("GTDom C Avg: {}, PredDom C Avg: {}".format(coarse_sigma_mean, mean_val_coarse))
print("MSE s: {}, PSNR s: {}".format(err, psnr), flush=True)
print("MSE sx: {}, PSNR sx: {}".format(errx, psnrx), flush=True)
print("MSE sy: {}, PSNR sy: {}".format(erry, psnry), flush=True)

slist = [scap, sxcap, sycap]
serr_list = [serr, sxerr, syerr]
err_list = [err, errx, erry]

plot_data_s(slist, serr_list, err_list, all_d, save_results_dir, epoch)

np.save(save_results_dir + 'loss_log.npy', s_loss_log)
np.save(save_results_dir + 'mse_s.npy', all_mse)


# In[ ]:


snet.load_weights(save_results_dir + 'snet_best.h5')

scap, sxcap, sycap = get_data()
err, psnr, serr = eval_error(scap, all_d[0]['Sigma'], all_d[0]['Omega'])
errx, psnrx, sxerr = eval_error(sxcap, all_d[0]['sx'], all_d[0]['Omega'])
erry, psnry, syerr = eval_error(sycap, all_d[0]['sy'], all_d[0]['Omega'])

coarse_sigma_mean = all_d[0]['Sigma'].mean()
coarse_sigmax_mean = all_d[0]['sx'].mean()
coarse_sigmay_mean = all_d[0]['sy'].mean()

mean_val_s_coarse = tf.reduce_mean(scap)
mean_val_sx_coarse = tf.reduce_mean(sxcap)
mean_val_sy_coarse = tf.reduce_mean(sycap)

mde_s = tf.square(coarse_sigma_mean - mean_val_s_coarse).numpy()
mde_sx = tf.square(coarse_sigmax_mean - mean_val_sx_coarse).numpy()
mde_sy = tf.square(coarse_sigmay_mean - mean_val_sy_coarse).numpy()

print('\nBest model results', flush=True)
print("\nMSE s: {}, PSNR s: {}, MDE s: {}".format(err, psnr, mde_s), flush=True)
print("MSE sx: {}, PSNR sx: {}, MDE sx: {}".format(errx, psnrx, mde_sx), flush=True)
print("MSE sy: {}, PSNR sy: {}, MDE sy: {}".format(erry, psnry, mde_sy), flush=True)

print("\nMSE s: {}, PSNR s: {}, MDE s: {}".format(np.round(err, 4), np.round(psnr, 2), np.round(mde_s, 4), flush=True))
print("MSE sx: {}, PSNR sx: {}, MDE sx: {}".format(np.round(errx, 4), np.round(psnrx, 2), mde_sx, flush=True))
print("MSE sy: {}, PSNR sy: {}, MDE sy: {}".format(np.round(erry, 4), np.round(psnry, 2), mde_sy, flush=True))


# In[ ]:




