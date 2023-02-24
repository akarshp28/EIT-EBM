import tensorflow as tf


def res_identity(x, filters): 
    x_skip = x
    f1, f2 = filters

    #first block 
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    #second block # bottleneck
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    # third block
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # add the input 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

def res_conv(x, s, filters):
    x_skip = x
    f1, f2 = filters
    
    # first block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    # second block
    x = tf.keras.layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)

    #third block
    x = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # shortcut 
    x_skip = tf.keras.layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_skip)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    # add 
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    return x

def resnet50():
    input_im = tf.keras.layers.Input(shape=(128, 128, 1))
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(input_im)
    
    # 1st stage
    x = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    #2nd stage 
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    
    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    
    x = tf.keras.layers.AveragePooling2D((4, 4), padding='same')(x)
    x_emb = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3, activation='softmax', kernel_initializer='he_normal')(x_emb)
    
    model = tf.keras.models.Model(input_im, [x_emb, x], name='Resnet50')
    return model