#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

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

def create_res_ebm_sigma(nef=32, act=tf.nn.elu, height=128, width=128):
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