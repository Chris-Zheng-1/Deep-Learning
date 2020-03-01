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
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils.np_utils import to_categorical 
mnist = input_data.read_data_sets('MNIST_data')


# In[2]:


mnist.train.images.shape
img=np.zeros((55000,28,28,1))


# In[3]:


for i in range(55000):
    img[i]=mnist.train.images[i].reshape(28,28,1)
img.shape
#载入训练集


# In[4]:


tt=np.zeros((10000,28,28,1))
for i in range(10000):
    tt[i]=mnist.test.images[i].reshape(28,28,1)


# In[5]:


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):

    x = Conv2D(nb_filter, kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
 
    identity = input
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


# In[6]:


model = resnet_18()#预览模型 ResNet


# In[7]:


from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-4, amsgrad=True), 
              metrics=['accuracy'])
#参数设置


# In[8]:


from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1)

datagen.fit(img)


# In[9]:


imgy=to_categorical(mnist.train.labels,num_classes=10)


# In[10]:


tt_label=to_categorical(mnist.test.labels,num_classes=10)


# In[11]:


gen=datagen.flow(img,imgy,batch_size=10000)


# In[12]:


for i in range(5):#这几个部分是在生成数据，包括一些旋转操作等等
    x,y=gen.next()
    model.fit(x,y,epochs=6,batch_size=80,shuffle=True,validation_data=(tt,tt_label))


# In[ ]:


model.save('test.h5')#把文件名改成自己喜欢的，比如test.h5


# In[60]:


#测试下效果
#载入一张图片
#反复运行此部分可以看到预测效果
import random
key=0
rdint = random.randint#生成随机数
num=rdint(1,1000)
ty=mnist.test.images[num]
plt.imshow(ty.reshape(28,28),cmap='gray')
ty=ty.reshape(1,28,28,1)
key= model.predict(ty)
key=np.argmax(key, axis=1)
print(key)

