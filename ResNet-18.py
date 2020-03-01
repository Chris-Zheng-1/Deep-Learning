#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, add, GlobalAvgPool2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


# In[4]:


mnist.train.images.shape
img=np.zeros((55000,28,28))


# In[5]:


x=np.zeros((10000,28,28))


# In[6]:


for i in range(10000):
    x[i]=mnist.test.images[i].reshape(28,28)
x.shape


# In[7]:


y=mnist.test.labels
y[1]


# In[8]:


x=x[:,:,:,np.newaxis
       ]


# In[9]:


for i in range(55000):
    img[i]=mnist.train.images[i].reshape(28,28)
img.shape


# In[10]:


img=img[:,:,:,np.newaxis
       ]


# In[11]:


img.shape


# In[12]:


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    """
    conv2d -> batch normalization -> relu activation
    """
    x = Conv2D(nb_filter, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


# In[13]:


def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
 
    identity = input
    # 如果维度不同，则使用1x1卷积进行调整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                           kernel_size=(1, 1),
                           strides=(stride_width, stride_height),
                           padding="valid",
                           kernel_regularizer=regularizers.l2(0.0001))(input)
 
    return add([identity, residual])
 
 
def basic_block(nb_filter, strides=(1, 1)):
    """
    基本的ResNet building block，适用于ResNet-18和ResNet-34.
    """
    def f(input):
 
        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn(conv1, nb_filter, kernel_size=(3, 3))
 
        return shortcut(input, residual)
 
    return f
 
 
def residual_block(nb_filter, repetitions, is_first_layer=False):

    def f(input):
        for i in range(repetitions):
            strides = (1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2)
            input = basic_block(nb_filter, strides)(input)
        return input
 
    return f
 
 
def resnet_18(input_shape=(28,28,1), nclass=10):
    input_ = Input(shape=input_shape)
 
    conv1 = conv2d_bn(input_, 64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
 
    conv2 = residual_block(64, 2, is_first_layer=True)(pool1)
    conv3 = residual_block(128, 2, is_first_layer=True)(conv2)
    conv4 = residual_block(256, 2, is_first_layer=True)(conv3)
    conv5 = residual_block(512, 2, is_first_layer=True)(conv4)
 
    pool2 = GlobalAvgPool2D()(conv5)
    output_ = Dense(nclass, activation='softmax')(pool2)
 
    model = Model(inputs=input_, outputs=output_)
    model.summary()
 
    return model


# In[14]:


model = resnet_18()


# In[15]:


from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-3, amsgrad=True), 
              metrics=['accuracy'])


# In[ ]:


model.fit(img,mnist.train.labels,epochs=1,batch_size=500,shuffle=True)


# In[16]:


model.fit(img,mnist.train.labels,epochs=1,batch_size=125,shuffle=True)


# In[21]:


from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-5, amsgrad=True), 
              metrics=['accuracy'])


# In[22]:


model.fit(img,mnist.train.labels,epochs=10,batch_size=200,shuffle=True,validation_data=(x,y))


# In[27]:


tx=np.zeros((2,28,28,1)
           )
tx[0]=img[12]
tx[1]=img[3123]


# In[29]:


model.save('test.h5')


# In[36]:


aa=model.predict(tx)
print(aa)
bb=np.argmax(aa,axis=1)
print(bb)


# In[37]:




