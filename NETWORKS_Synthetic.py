from __future__ import absolute_import, print_function

import json
import os
import sys
sys.path.append('/home/xuchen/Desktop/CVPR20_2.0/ALL_IN_EXP/LIBS')
sys.path.append('/home/xuchen/Desktop/CVPR20_2.0/ALL_IN_EXP/LIBS/neuron')
sys.path.append('/home/xuchen/Desktop/CVPR20_2.0/ALL_IN_EXP/LIBS/pynd-lib')
sys.path.append('/home/xuchen/Desktop/CVPR20_2.0/ALL_IN_EXP/LIBS/pytools-lib')

import cv2
import keras.models as models
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import (Concatenate, Conv2D, Conv2DTranspose, Input, Lambda,
                          concatenate, PReLU, LeakyReLU)
from keras.layers.convolutional import (AveragePooling2D, Conv2D, Cropping2D,
                                        MaxPooling2D, UpSampling2D)
from keras.layers.core import (Activation, Dense, Dropout, Flatten, Layer,
                               Permute, Reshape)
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.activations import hard_sigmoid, relu
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from grid import batch_mgrid
import LOSS
from keras import losses
from STN import (Mapping_each_aff_para,
                 batch_displacement_warp2d, batch_displacement_warp2d_tf,
                 warp_2d_layer_output_shape)

K.set_image_dim_ordering('th')
import LAYERS


def RAN_Affine_Block(Conv_f, dpr, x):

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation(lambda x : relu(x, alpha = 0.1))(x)
        x = Dropout(dpr, seed = 1203)(x)

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation(lambda x : relu(x, alpha = 0.1))(x)
        x = Dropout(dpr, seed = 1203)(x)

        x = AveragePooling2D(pool_size=(2, 2))(x)

        return x

def RAN_NL_DownSampling_Block(Conv_f, dpr, x):

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation(lambda x : relu(x, alpha = 0.1))(x)
        x = Dropout(dpr, seed = 1203)(x)

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x0 = Activation(lambda x : relu(x, alpha = 0.1))(x)

        x1 = AveragePooling2D(pool_size=(2, 2))(x0)

        return x0, x1


def RAN_NL_UpSampling_Block(Conv_f, dpr, x1, x2):

        x = Concatenate(axis=1)([UpSampling2D(size=(2, 2))(x1), x2])

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation(lambda x : relu(x, alpha = 0.1))(x)
        x = Dropout(dpr, seed = 1203)(x)

        x = Conv2D(Conv_f, (3, 3), padding='same', data_format='channels_first')(x)
        x = BatchNormalization()(x)
        x = Activation(lambda x : relu(x, alpha = 0.1))(x)

        return x

def STN_RAN(        
                n_ch, patch_height, patch_width,
                optimizer,
                Test_AFF_NL,
                Test_AFF_Loss,
                Test_NL_Loss,
                Affine_Activation_type,  
                NL_Activation_type,
                Affine_Loss_type, 
                NL_Loss_type, 
        ):
             
        src = Input((1, patch_height, patch_width), name = 'src')
        tgt = Input((1, patch_height, patch_width), name = 'tgt')
        dpr = 0.2
        src_1 = RAN_Affine_Block(64, dpr, src)
        tgt_1 = RAN_Affine_Block(64, dpr, tgt)

        src_2 = RAN_Affine_Block(64, dpr, src_1)
        tgt_2 = RAN_Affine_Block(64, dpr, tgt_1)

        src_3 = RAN_Affine_Block(32, dpr, src_2)
        tgt_3 = RAN_Affine_Block(32, dpr, tgt_2)

        src_4 = RAN_Affine_Block(16, dpr, src_3)
        tgt_4 = RAN_Affine_Block(16, dpr, tgt_3)

        src_5 = RAN_Affine_Block(8, dpr, src_4)
        tgt_5 = RAN_Affine_Block(8, dpr, tgt_4)

        src_6 = RAN_Affine_Block(8, dpr, src_5)
        tgt_6 = RAN_Affine_Block(8, dpr, tgt_5)

        src_7 = RAN_Affine_Block(8, dpr, src_6)
        tgt_7 = RAN_Affine_Block(8, dpr, tgt_6)

        src_8 = RAN_Affine_Block(8, dpr, src_7)
        tgt_8 = RAN_Affine_Block(8, dpr, tgt_7)

        conc_src_tgt = Concatenate( axis = 1 )([src_8, tgt_8])

        conc_src_tgt_0 = Conv2D(6, (3, 3), padding='same', data_format = 'channels_first')(conc_src_tgt)
        conc_src_tgt_0 = BatchNormalization()(conc_src_tgt_0)
        conc_src_tgt_0 = Activation((lambda x : relu(x, alpha = 0.1)))(conc_src_tgt_0)


        conc_src_tgt_Flatten = Flatten()(conc_src_tgt_0)
        Mapped_Affine_Para = Dense(6, name = 'AFF_STN')(conc_src_tgt_Flatten)
        Mapped_Affine_Para = BatchNormalization()(Mapped_Affine_Para)
        Mapped_Affine_Para = Activation((lambda x : relu(x, alpha = 0.1)))(Mapped_Affine_Para)

        Affine_Warped = Lambda(    lambda x: LAYERS.affine_flow(x), output_shape = LAYERS.affine_flow_output_shape, 
                                        name = 'Affine_Warping')([src, Mapped_Affine_Para])


        inputs = Concatenate( axis = 1 )([Affine_Warped, tgt])


        conv1, pool1 = RAN_NL_DownSampling_Block(32, dpr, inputs)
        conv2, pool2 = RAN_NL_DownSampling_Block(16, dpr, pool1)
        conv3, pool3 = RAN_NL_DownSampling_Block(32, dpr, pool2)
        conv4, pool4 = RAN_NL_DownSampling_Block(64, dpr, pool3)
        conv5, pool5 = RAN_NL_DownSampling_Block(128, dpr, pool4)

        # print(" ################################# conv5: "+str(conv5.shape))
        # print(" ################################# pool4: "+str(pool4.shape))
        
        conv6 = RAN_NL_UpSampling_Block(64, dpr, conv5, conv4)
        conv7 = RAN_NL_UpSampling_Block(32, dpr, conv6, conv3)
        conv8 = RAN_NL_UpSampling_Block(16, dpr, conv7, conv2)
        conv9 = RAN_NL_UpSampling_Block(16, dpr, conv8, conv1)

        # conv9 = 
        # conv10 = RAN_NL_DownSampling_Block(8, dpr, conv9)
        # conv11 = RAN_NL_DownSampling_Block(4, dpr, conv10)
        # print(" ################################# conv11: "+str(conv11.shape))

        max_value = 1.0
        Threshold = 0.0

        if NL_Activation_type == 'Sigmoid':
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Sigmoid',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation('sigmoid')(Non_linear_flow)

        elif NL_Activation_type == 'Hard_Sigmoid':
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Hard_Sigmoid',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation('hard_sigmoid')(Non_linear_flow)

        elif NL_Activation_type == 'Large_Hard_Sigmoid':

                def Large_Hard_Sigmoid(x):
                        return tf.where(        condition = x < -5.0, 
                                                x = 0.0 * x,
                                                y = tf.where(
                                                                condition = x > 5.0, 
                                                                x = 0.0 * x + 1.0,
                                                                y = 0.1 * x + 0.5
                                                                )
                                                )


                # Affine_Para_0 = Lambda(   lambda x: tf.where(   condition = x < 0.0, 
                #                                                                 x = tf.multiply(alpha, x),
                #                                                                 y = tf.where( condition = x > 1.0, 
                #                                                                                         x = tf.log(x - 1.0) + y_max,
                #                                                                                         y = (y_max - y_min) * x + y_min)
                #                                                             ),



                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Large_Hard_Sigmoid',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : Large_Hard_Sigmoid(x))(Non_linear_flow)

        elif NL_Activation_type == 'Leaky_Hard_Sigmoid':
                def Leaky_Hard_Sigmoid(x):
                        return tf.where(        condition = x < -2.5, 
                                                x = 0.1 * x,
                                                y = hard_sigmoid(x)
                                                )
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Leaky_Hard_Sigmoid',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : Leaky_Hard_Sigmoid(x))(Non_linear_flow)

        elif NL_Activation_type == 'Leaky_ReLU':
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Leaky_ReLU',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : relu(x, alpha = 0.1, max_value = None, threshold = Threshold))(Non_linear_flow)

        elif NL_Activation_type == 'Cliped_ReLU':
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Clipped_ReLU',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : relu(x, alpha = 0.0, max_value = max_value, threshold = Threshold))(Non_linear_flow)

        elif NL_Activation_type == 'Leaky_Cliped_ReLU':
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Clipped_Leaky_ReLU',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : relu(x, alpha = 0.1, max_value = max_value, threshold = Threshold))(Non_linear_flow)

        elif NL_Activation_type == 'Large_Hard_Sigmoid-10':
                def Large_Hard_Sigmoid(x):
                        return tf.where(        condition = x < -2.5, 
                                                x = 0.0 * x - 0.1,
                                                y = tf.where(
                                                                condition = x > 2.5, 
                                                                x = 0.0 * x + 1.0,
                                                                y = 0.4 * x 
                                                                )
                                                )
                Non_linear_flow = Conv2D(2, (3, 3), padding='same', data_format='channels_first', name = 'NL_Large_Hard_Sigmoid-10',
                                                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(conv9)
                Non_linear_flow = BatchNormalization()(Non_linear_flow)
                Non_linear_flow = Activation(lambda x : Large_Hard_Sigmoid(x))(Non_linear_flow)
                
        Third_flow = Lambda(lambda x: x * 0, name = '3rd_Zero_flow')(src)

        Show_Non_linear_flow = Concatenate(axis = 1)([Non_linear_flow, Third_flow])



        Non_linear_flow = Lambda(lambda x: 2*x-1, name = 'mapping_flow')(Non_linear_flow)




        Non_linear_Warped = Lambda( lambda x: batch_displacement_warp2d(x), output_shape = warp_2d_layer_output_shape, 
                                        name = 'Non_linear_Warping')([Affine_Warped, Non_linear_flow])


        model = Model(inputs=[src, tgt], outputs=[Affine_Warped, Non_linear_Warped])

        model.input_tgt = tgt
        model.output_Non_linear_flow = Show_Non_linear_flow
        model.output_Affine_Warped = Affine_Warped
        model.output_Non_linear_Warped = Non_linear_Warped
        model.output_Mapped_Affine_Para = Mapped_Affine_Para


        def NMI_Loss(y_true, y_pred):
                # Normalized Mutual Information
                y_true = K.clip(y_true, K.epsilon(), 1)
                y_pred = K.clip(y_pred, K.epsilon(), 1)
                KL_loss = K.sum(y_true * K.log(y_true / y_pred), axis = 1) # Kullback_Leibler_Divergence
                entropy_y_true = K.sum(y_true * K.log(y_true))
                entropy_y_pred = K.sum(y_pred * K.log(y_pred))
                # NMI_loss = KL_loss / K.sqrt(entropy_y_true * entropy_y_pred)
                NMI_loss = 2 * KL_loss / (entropy_y_true * entropy_y_pred + 0.0001)
                return NMI_loss

        def NCC_Score(y_true, y_pred):
                m_y_true, m_y_pred = K.mean(y_true), K.mean(y_pred)
                diff_y_true = (y_true - m_y_true)
                diff_y_pred = (y_pred - m_y_pred)
                num = K.sum(diff_y_true * diff_y_pred)
                den0 = K.sum(K.square(diff_y_true)) * K.sum(K.square(diff_y_pred))
                den = K.sqrt(den0 + 0.000001)
                NCC = num / ( den + 0.000001)
                return NCC

        def NCC_Loss(y_true, y_pred):
                # Normalized 2-D cross-correlation
                return 1 - NCC_Score(y_true, y_pred)

        def Huber_Loss(y_true, y_pred):
                clip_delta=1.0
                error = y_true - y_pred
                cond  = tf.keras.backend.abs(error) < clip_delta
                squared_loss = 0.5 * tf.keras.backend.square(error)
                linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
                return tf.where(cond, squared_loss, linear_loss)

        def SSD_Loss(y_true, y_pred):
                # Sum of Squared Differences
                SSD_loss = K.mean((K.square(K.abs(y_true - y_pred))), axis=1) #/(256*256)
                return SSD_loss

        def Affine_Loss(tgt, Affine_Warped, loss_name):
                y_true = tgt
                y_pred = Affine_Warped
                if loss_name == "NCC_Loss":
                        def loss(y_true, y_pred):
                                ncc_loss = NCC_Loss(y_true, y_pred)
                                return ncc_loss
                if loss_name == "NMI_Loss":
                        def loss(y_true, y_pred):
                                nmi_loss = NMI_Loss(y_true, y_pred)
                                return nmi_loss
                if loss_name == "Huber_Loss":
                        def loss(y_true, y_pred):
                                Huber_loss = Huber_Loss(y_true, y_pred)
                                return Huber_loss
                if loss_name == "SSD_Loss":
                        def loss(y_true, y_pred):
                                ssd_loss = SSD_Loss(y_true, y_pred)/256/256
                                return ssd_loss
                return loss

        def my_flow_smooth_loss(y_pred):
                y_pred = y_pred[:,0:2,:,:]
                x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
                y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]
                delta_x = x[:,:,1:,:-2]**2
                delta_y = y[:,:,:-2,1:]**2
                delta_u = K.abs(delta_x + delta_y) 
                epsilon = 0.00000001
                delta = K.mean(K.sqrt(delta_u + epsilon))
                return delta

        def Non_linear_Loss(tgt, Non_linear_Warped, flow, loss_name):
                y_true = tgt
                y_pred = Non_linear_Warped
                w = 1
                smooth_loss = my_flow_smooth_loss(flow)
                if loss_name == "NCC_Loss":
                        def loss(y_true, y_pred):
                                ncc_loss = NCC_Loss(y_true, y_pred) + w * smooth_loss 
                                return ncc_loss
                if loss_name == "NMI_Loss":
                        def loss(y_true, y_pred):
                                nmi_loss = NMI_Loss(y_true, y_pred) + w * smooth_loss 
                                return nmi_loss
                if loss_name == "Huber_Loss":
                        def loss(y_true, y_pred):
                                Huber_loss = Huber_Loss(y_true, y_pred) + w * smooth_loss 
                                return Huber_loss
                if loss_name == "SSD_Loss":
                        def loss(y_true, y_pred):
                                ssd_loss = SSD_Loss(y_true, y_pred)/256/256 + w * smooth_loss 
                                return ssd_loss
                return loss 



        if Test_AFF_Loss == 1:

                model.compile(
                                optimizer = optimizer, 
                                loss= [ Affine_Loss(tgt, Affine_Warped, Affine_Loss_type),
                                        Non_linear_Loss(Affine_Warped, Non_linear_Warped, Non_linear_flow, NL_Loss_type)],
                                loss_weights = [1, 0],
                                metrics=[NCC_Score],
                                )

        elif Test_NL_Loss == 1:

                model.compile(
                                optimizer = optimizer, 
                                loss= [ Affine_Loss(tgt, Affine_Warped, Affine_Loss_type),
                                        Non_linear_Loss(Affine_Warped, Non_linear_Warped, Non_linear_flow, NL_Loss_type)],
                                loss_weights = [0, 1],
                                metrics=[NCC_Score],
                                )

        elif Test_AFF_NL == 1:

                model.compile(
                                optimizer = optimizer, 
                                loss= [ Affine_Loss(tgt, Affine_Warped, Affine_Loss_type),
                                        Non_linear_Loss(Affine_Warped, Non_linear_Warped, Non_linear_flow, NL_Loss_type)],
                                loss_weights = [1, 1],
                                metrics=[NCC_Score],
                                )
      
        return model