#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from utils_ebm import make_dirs, plot_grid


# In[2]:


exp = 0
exp_name = 'dsm'
save_results_dir = '/scratch/apokkunu/eit_iclr_full/' + exp_name + '/trial_' + str(exp) + '/'
make_dirs(save_results_dir)
print('\n', save_results_dir, flush=True)


# In[3]:


with open('../../dataset/multisigma_ebm_128_no_circle_v2.pickle', 'rb') as handle:
    all_data = pickle.load(handle)

mask = np.load('./omega_mask.npy').astype(np.float32)

xtrain = mask * all_data['xtrain']
xval = mask * all_data['xval']
xtest = mask * all_data['xtest']
del all_data

xtrain = np.expand_dims(xtrain, -1)
xval = np.expand_dims(xval, -1)
xtest = np.expand_dims(xtest, -1)

max_sigma = np.max(xtrain)
xtrain = xtrain/max_sigma
xval = xval/max_sigma
xtest = xtest/max_sigma

xtrain = xtrain.astype(np.float32)
xval = xval.astype(np.float32)
xtest = xtest.astype(np.float32)

iclr = False # experiments for low-data EBM

if iclr:
    p_mode = int(exp - 1)
    percent_data = [0.01, 0.05, 0.1, 0.2, 0.5]
    np.random.shuffle(xtrain)
    xtrain = xtrain[:round(xtrain.shape[0] * percent_data[p_mode])]

print('X: ', xtrain.shape, xval.shape, xtest.shape, flush=True)


# In[4]:


if iclr:
    batch_size = 16
else:
    batch_size = 128

train_dataset = tf.data.Dataset.from_tensor_slices(xtrain).shuffle(xtrain.shape[0])
train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=10000)
test_dataset = tf.data.Dataset.from_tensor_slices(xtest).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=10000)
val_dataset = tf.data.Dataset.from_tensor_slices(xval).batch(512, num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=10000)


# In[5]:


def ConvResBlock(xin, f=32, g=32, resize=False, act=tf.nn.elu):
    if not resize:
        x = tf.keras.layers.Conv2D(f, kernel_size=3, strides=1, padding='same')(xin)
        x = tfa.layers.GroupNormalization(groups=g)(x)
        x = tf.keras.layers.Activation(act)(x)
        x = tf.keras.layers.Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=g)(x)
        x = tf.keras.layers.Add()([xin, x])
    else:
        x = tf.keras.layers.Conv2D(f, kernel_size=3, strides=2, padding='same')(xin)
        x = tfa.layers.GroupNormalization(groups=g)(x)
        x = tf.keras.layers.Activation(act)(x)
        x = tf.keras.layers.Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
        x = tfa.layers.GroupNormalization(groups=g)(x)
        res_x = tf.keras.layers.Conv2D(f, kernel_size=3, strides=2, padding='same')(xin)
        x = tf.keras.layers.Add()([x, res_x])
    final_act = tf.keras.layers.Activation(act)(x)
    return final_act

def create_res_ebm(nef=32, act=tf.nn.elu, height=128, width=128):
    input_layer = tf.keras.layers.Input(shape=(height*width))
    label_layer = tf.keras.layers.Input(shape=(1))
    
    x = tf.keras.layers.Reshape([height, width, 1])(input_layer)
    x = 2 * x - 1
    
    x = tf.keras.layers.Conv2D(nef, kernel_size=3, strides=1, padding='same')(x)
    x = ConvResBlock(x, nef, resize=False, act=act)
    x = ConvResBlock(x, nef, resize=False, act=act)
    
    x = ConvResBlock(x, nef*2, resize=True, act=act)
    x = ConvResBlock(x, nef*2, resize=False, act=act)
    x = ConvResBlock(x, nef*2, resize=True, act=act)
    x = ConvResBlock(x, nef*2, resize=False, act=act)
    
    x = ConvResBlock(x, nef*4, resize=True, act=act)
    x = ConvResBlock(x, nef*4, resize=False, act=act)
    x = ConvResBlock(x, nef*4, resize=True, act=act)
    x = ConvResBlock(x, nef*4, resize=False, act=act)
    
    x = ConvResBlock(x, nef*8, resize=True, act=act)
    x = ConvResBlock(x, nef*8, resize=False, act=act)
    x = ConvResBlock(x, nef*8, resize=True, act=act)
    x = ConvResBlock(x, nef*8, resize=False, act=act)
    
    x = ConvResBlock(x, nef*16, resize=True, act=act)
    x = ConvResBlock(x, nef*16, resize=False, act=act)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x, label_layer])
    x = tf.keras.layers.Dense(256, activation=act)(x)
    x = tf.keras.layers.Dense(1, use_bias=False)(x)
    x = tf.squeeze(x)
    
    model = tf.keras.models.Model([input_layer, label_layer], x, name='EBM')
    return model

# # Denoising Score Matching Equation 2 --> Generative modelling Song 2019

# ### Annealed Variant

# In[11]:


@tf.function
def anneal_dsm_train_step(x, train, std, lbl, anneal_power=2.0):
    x = x / 256. * 255. + tf.random.uniform(x.shape) / 256.
    x = tf.reshape(x, [x.shape[0], -1])
    
    std = tf.reshape(tf.convert_to_tensor(std), [-1, 1])
    lbl = tf.reshape(tf.convert_to_tensor(lbl), [-1, 1])
    used_sigma = tf.math.pow(std, anneal_power)
    
    if train:
        with tf.GradientTape() as tape:
            vector = tf.random.normal(x.shape) * std
            perturbed_inputs = x + vector
            logp = tf.reduce_sum(-ebm([perturbed_inputs, lbl]))
            dlogp = tf.gradients(logp, perturbed_inputs)[0] * used_sigma
            norm_val = tf.reduce_sum(tf.square(dlogp + vector), axis=-1)
            loss = 0.5 * tf.reduce_mean(norm_val)
        gradients = tape.gradient(loss, ebm.trainable_variables)
        optimizer.apply_gradients(zip(gradients, ebm.trainable_variables))
    else:
        ebm.trainable = False
        vector = tf.random.normal(x.shape) * std
        perturbed_inputs = x + vector
        logp = tf.reduce_sum(-ebm([perturbed_inputs, lbl]))
        dlogp = tf.gradients(logp, perturbed_inputs)[0] * used_sigma
        norm_val = tf.reduce_sum(tf.square(dlogp + vector), axis=-1)
        loss = 0.5 * tf.reduce_mean(norm_val)
    return loss


# In[12]:


total_epochs = 2000
init_lr = 0.0001
lr_decay = False

early_stopping = False
patience = 100

decay_every = 100
decay_rate = 0.9
optimizer = tf.keras.optimizers.Adam(init_lr)

anneal = True
noise_max = 2
noise_min = 0.01
noise_levels = 20
sigmas = np.exp(np.linspace(np.log(noise_max), np.log(noise_min), noise_levels))

act = tf.nn.elu
ebm = create_res_ebm()
ebm.summary()


# In[13]:


template = "\nbatch_size: {}, total_epochs: {}, decay_every: {}, init_lr: {}, lr_decay: {}, opt: {}"
print(template.format(batch_size, total_epochs, decay_every, init_lr, lr_decay, 'Adam'), flush=True)

if iclr:
    template = "\nact: {}, method: {}, anneal: {}, Train Data %: {}"
    print(template.format(act, exp_name, anneal, percent_data[p_mode]), flush=True)
else:
    template = "\nact: {}, method: {}, anneal: {}"
    print(template.format(act, exp_name, anneal), flush=True)

template = "\nearly_stopping: {}, patience: {}, noise_max: {}, noise_min: {}, noise_levels: {}"
print(template.format(early_stopping, patience, noise_max, noise_min, noise_levels), flush=True)


# # Training Loop

# In[14]:


def fit(tfds, sigmas, train=False):
    total_loss = 0
    for xbatch in tfds:
        labels = np.random.randint(0, len(sigmas), (xbatch.shape[0])).astype('int32')
        sigma = sigmas[labels].astype('float32')
        total_loss += anneal_dsm_train_step(xbatch, train, sigma, labels)
    return total_loss.numpy()/len(tfds)


# In[15]:


@tf.function
def compute_grad(x, labels):
    en = tf.reduce_sum(-ebm([x, labels]))
    g = tf.gradients(en, x)[0]
    return en, g

def anneal_inference(batch, iiter, irate, noise_rate, clip=True, add_noise=True, verbose=0, height=128, width=128):
    y = tf.random.uniform([batch, height*width])
    
    for c, sigma in enumerate(sigmas):
        
        labels = tf.ones([y.shape[0], 1]) * c
        step_size = irate * (sigma / sigmas[-1]) ** 2
        print('sigma: {}, step size: {}'.format(sigma, step_size), flush=True)
        
        for i in range(iiter):
            
            if i == iiter-1:
                add_noise = False
            
            if clip:
                y = tf.clip_by_value(y, clip_value_min=0, clip_value_max=1)
            else:
                y = tf.nn.sigmoid(y)
                
            en, g = compute_grad(y, labels)
            
            if add_noise:
                noise = tf.random.normal(y.shape) * np.sqrt(irate * noise_rate)
                y = y + irate * g + noise
            else:
                y = y + irate * g
            
            if verbose > 0:
                gn = tf.reduce_sum(tf.square(g), axis=-1)
                print("step: {}, en: {}, g: {}, en b: {}".format(i, en[0], gn[0], tf.reduce_mean(en)), flush=True)
    
    if clip:
        y = tf.clip_by_value(y, clip_value_min=0, clip_value_max=1)
    else:
        y = tf.nn.sigmoid(y)
    
    return y.numpy().reshape(-1, 128, 128)

def anneal_mask_inference(batch, iiter, irate, noise_rate, clip=True, add_noise=True, verbose=0, height=128, width=128):
    y = tf.random.uniform([batch, height, width, 1])
    masktile = tf.tile(tf.expand_dims(tf.expand_dims(mask, axis=0), -1), [batch, 1, 1, 1])
    y = y * masktile
    
    for c, sigma in enumerate(sigmas):

        labels = tf.ones([batch, 1]) * c
        step_size = irate * (sigma / sigmas[-1]) ** 2
        print('sigma: {}, step size: {}'.format(sigma, step_size), flush=True)
        
        for i in range(iiter):

            if i == iiter-1:
                add_noise = False

            if clip:
                y = tf.clip_by_value(y, clip_value_min=0, clip_value_max=1)
            else:
                y = tf.nn.sigmoid(y)

            y_re = tf.reshape(y, [batch, -1])
            y = tf.reshape(y_re, [batch, height, width, 1])

            en, g = compute_grad(y_re, labels)
            g = tf.reshape(g, [batch, height, width, 1])

            if add_noise:
                noise = tf.random.normal(y.shape) * np.sqrt(irate * noise_rate)
                y = y + irate * g + noise
            else:
                y = y + irate * g

            y = y * masktile

            if verbose > 0:
                gn = tf.reduce_sum(tf.square(g), axis=-1)
                print("step: {}, en: {}, g: {}, en b: {}".format(i, en[0], gn[0], tf.reduce_mean(en)), flush=True)
    
    if clip:
        y = tf.clip_by_value(y, clip_value_min=0, clip_value_max=1)
    else:
        y = tf.nn.sigmoid(y)

    return y.numpy().reshape(-1, 128, 128)



# In[ ]:


best_loss = 1e10
best_epoch = 0

for epoch in range(total_epochs):
    
    # annealed training
    train_loss = fit(train_dataset, sigmas, True)
    test_loss = fit(test_dataset, sigmas)
    val_loss = fit(val_dataset, sigmas)
    
    if val_loss <= best_loss:
        best_loss = val_loss
        best_epoch = epoch
    
    if epoch % 100 == 0:
        ebm.save_weights(save_results_dir + 'ebm_best_' + str(epoch) + '.h5')
        
        y = anneal_inference(batch=25, iiter=50, irate=0.0018, noise_rate=2)
        plot_grid(y*max_sigma, 5, 5, 10, save_results_dir, epoch, 'normal')
        
        y = anneal_mask_inference(batch=25, iiter=50, irate=0.0002, noise_rate=2)
        plot_grid(y*max_sigma, 5, 5, 10, save_results_dir, epoch, 'masked')
        
        print("\nEpoch: {}, Best Epoch: {}".format(epoch, best_epoch), flush=True)
        print("Train Loss: {}, Val Loss: {}, Test Loss: {}".format(train_loss, val_loss, test_loss), flush=True)
    
    if early_stopping and (epoch - best_epoch) >= patience:
        print('\nEarly Stopped', flush=True)
        break


# In[ ]:


ebm.save_weights(save_results_dir + 'ebm_best_' + str(epoch) + '.h5')

y = anneal_inference(batch=100, iiter=60, irate=0.0018, noise_rate=2)
plot_grid(y*max_sigma, 10, 10, 10, save_results_dir, epoch, 'final')

y = anneal_mask_inference(batch=100, iiter=60, irate=0.0018, noise_rate=2)
plot_grid(y*max_sigma, 5, 5, 10, save_results_dir, epoch, 'final_masked')
