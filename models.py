
import os
import sys
import random
sys.path.append('../libs')
sys.path.append('./neuron')
sys.path.append('./pynd-lib')
sys.path.append('./pytools-lib')


import tensorflow as tf
import keras.backend as K
from keras.layers import (Layer, Add, Concatenate, Conv2D, Conv2DTranspose, Input, Lambda, concatenate, PReLU, LeakyReLU,GlobalAveragePooling2D)
from keras.layers import (Layer, Add, Concatenate, Conv3D, Conv3DTranspose, Input, Lambda, concatenate, PReLU, LeakyReLU,GlobalAveragePooling3D)
from keras.layers.normalization import BatchNormalization
from keras.layers.core import (Activation, Dense, Dropout, Flatten, Layer, Permute, Reshape)
from keras.layers.convolutional import (AveragePooling2D, Conv2D, Cropping2D, MaxPooling2D, UpSampling2D, MaxPooling3D, AveragePooling3D)
from keras.activations import hard_sigmoid, relu
from keras.models import Model

import numpy as np

import layers
import losses
import metrics


def affine_Loss_zoo(tgt, Affine_Warped, loss_name):
                y_true = tgt
                y_pred = Affine_Warped
                if loss_name == "NCCLoss":
                        def loss(y_true, y_pred):
                                ncc_loss = losses.NCC_loss(y_true, y_pred)
                                return ncc_loss
                # elif loss_name == "LNCCLoss":
                #         def loss(y_true, y_pred):
                #                 ncc_loss = losses.LNCC_loss(y_true, y_pred)
                #                 return ncc_loss
                # elif loss_name == "DCLoss":
                #         def loss(y_true, y_pred):
                #                 dc_loss = losses.DC_loss(y_true, y_pred)
                #                 return dc_loss
                # if loss_name == "NMI_Loss":
                #         def loss(y_true, y_pred):
                #                 nmi_loss = NMI_Loss(y_true, y_pred)
                #                 return nmi_loss
                # if loss_name == "Huber_Loss":
                #         def loss(y_true, y_pred):
                #                 Huber_loss = Huber_Loss(y_true, y_pred)
                #                 return Huber_loss
                # if loss_name == "SSD_Loss":
                #         def loss(y_true, y_pred):
                #                 ssd_loss = SSD_Loss(y_true, y_pred)/256/256
                #                 return ssd_loss
                return loss


def DLIR2D(exp_name, optimizer, PASTA): 
             
        def Conv2dBlock(filters, x):

                dpr = 0.2
                x = Conv2D(
                        filters, (3, 3), 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 1)(x)

                x = Conv2D(
                        filters, (3, 3), 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 2)(x)

                x = AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)

                return x

        src = Input((1, 64, 64), name = 'src')
        tgt = Input((1, 64, 64), name = 'tgt')

        x1 = src
        x2 = tgt

        num_channels = [ 64, 64, 32, 16, 8, 8]

        for level in num_channels:
                # print(level)
                x1 = Conv2dBlock(level, x1)
                x2 = Conv2dBlock(level, x2)

        
        x = Concatenate( axis = 1 )([x1, x2])

        if PASTA == True:

                # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
                # Rotation_2D_Range = np.arange(-30,30.01,10)
                # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
                # Scale_2D_Range = np.arange(0.90,1.101,0.05)

                x = Flatten()(x)

                transformation_params = Dense(7)(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(7):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 2:
                                # y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 3  or i == 4:
                                y_max, y_min = 0.101, -0.101
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 5  or i == 6:
                                y_max, y_min = 1.101, 0.901
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_7 = Concatenate( axis = 1, name = 'PASTA_7')(branch_outputs)

                AFF_6 = Lambda(  lambda x:layers.Combining_Affine_Para(x), name = 'AFF_6')([src, PASTA_7])

        else:
                initial_affine_matrix = [
                                                1.0, 0.0, 0.0, 
                                                0.0, 1.0, 0.0]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                x = Flatten()(x)

                AFF_6 = Dense(6, bias_initializer=initial_affine_matrix, name = 'AFF_6')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow(x), output_shape = layers.affine_flow_output_shape, name = 'Affine_Warpped')([src, AFF_6])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_6 = AFF_6

        if PASTA == True:
                model.PASTA_7 = PASTA_7

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.NCC_Score]
                        )

        return model

def GlobalNet2D(exp_name, optimizer, PASTA):
             
        def Conv2dBlock(filters, x):

                x = Conv2D(
                        filters, (3, 3), 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)

                x = Activation('relu')(x)

                return x

        src = Input((1, 64, 64), name = 'src')
        tgt = Input((1, 64, 64), name = 'tgt')
        Inputs = Concatenate( axis = 1 )([src, tgt])

        num_channels = [ 64, 64, 32, 32, 16, 8]
        extract_max_level = len(num_channels)

        x = Inputs

        for level in range(extract_max_level):  

                filters = num_channels[level]

                x = Conv2dBlock(filters, x)

                x = MaxPooling2D(
                                pool_size=(2, 2), 
                                strides=(2, 2),
                                data_format='channels_first')(x)


        x = Conv2dBlock(num_channels[-1], x)

        x = Flatten(data_format = 'channels_first')(x)

        if PASTA == True:

                # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
                # Rotation_2D_Range = np.arange(-30,30.01,10)
                # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
                # Scale_2D_Range = np.arange(0.90,1.101,0.05)

                initial_transformation_params = [
                        0.0, 0.0, 
                        0.0, 
                        0.0, 0.0,
                        1.0, 1.0]

                initial_transformation_params = tf.constant_initializer(value=initial_transformation_params)

                transformation_params = Dense(7, bias_initializer = initial_transformation_params, name = 'transformation_params')(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(7):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 2:
                                # y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 3  or i == 4:
                                y_max, y_min = 0.101, -0.101
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 5  or i == 6:
                                y_max, y_min = 1.101, 0.901
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_7 = Concatenate( axis = 1, name = 'PASTA_7')(branch_outputs)

                AFF_6 = Lambda(  lambda x:layers.Combining_Affine_Para(x), name = 'AFF_6')([src, PASTA_7])

                

        else:
                initial_affine_matrix = [
                1.0, 0.0, 0.0, 
                0.0, 1.0, 0.0]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                AFF_6 = Dense(6, bias_initializer=initial_affine_matrix, name = 'AFF_6')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow(x), output_shape = layers.affine_flow_output_shape, name = 'Affine_Warpped')([src, AFF_6])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_6 = AFF_6

        if PASTA == True:
                model.PASTA_7 = PASTA_7

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.NCC_Score, metrics.NCC_Score]
                        )

        return model

def GlobalNet3D(exp_name, optimizer, PASTA):
             
        def Conv3dBlock(filters, x):

                x = Conv3D(
                        filters, kernel_size=3, 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)

                x = Activation('relu')(x)

                return x

        # src = Input((128, 128, 128), name = 'src')
        # tgt = Input((128, 128, 128), name = 'tgt')
        src = Input((64, 64, 64), name = 'src')
        tgt = Input((64, 64, 64), name = 'tgt')
        src1 = Lambda(lambda x: K.expand_dims(x, axis=1))(src)
        tgt1 = Lambda(lambda x: K.expand_dims(x, axis=1))(tgt)

        Inputs = Concatenate( axis = 1 )([src1, tgt1])

        num_channels = [ 64, 32, 32, 32, 16, 16]
        extract_max_level = len(num_channels)

        x = Inputs

        for level in range(extract_max_level):  

                filters = num_channels[level]

                x = Conv3dBlock(filters, x)

                x = MaxPooling3D(pool_size=2,
                                data_format='channels_first')(x)


        x = Conv3dBlock(num_channels[-1], x)

        x = Flatten(data_format = 'channels_first')(x)

        if PASTA == True:

                # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
                # Rotation_2D_Range = np.arange(-30,30.01,10)
                # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
                # Scale_2D_Range = np.arange(0.90,1.101,0.05)

                initial_transformation_params = [
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0, 1.0, 1.0]

                initial_transformation_params = tf.constant_initializer(value=initial_transformation_params)

                transformation_params = Dense(15, bias_initializer = initial_transformation_params, name = 'transformation_params')(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(15):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1 or i == 2:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 3 or i == 4 or i == 5:
                                y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                # y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5 and i < 12:
                                y_max, y_min = 0.051, -0.051
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5:
                                y_max, y_min = 1.051, 0.951
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_15 = Concatenate( axis = 1, name = 'PASTA_15')(branch_outputs)

                AFF_12 = Lambda(  lambda x:layers.Combining_Affine_Para3D(x), name = 'AFF_12')([src, PASTA_15])

                

        else:
                initial_affine_matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                ]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                AFF_12 = Dense(12, bias_initializer=initial_affine_matrix, name = 'AFF_12')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow_3D(x), output_shape = layers.affine_flow_3D_output_shape, name = 'Affine_Warpped')([src1, AFF_12])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_12 = AFF_12

        if PASTA == True:
                model.PASTA_15 = PASTA_15

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )
        elif "DCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )

        return model

def CroStchNet2D(exp_name, optimizer, unit_num, PASTA):
             
        def Conv2dBlock(filters, x):
                
                dpr = 0.2
                x = Conv2D(
                        filters, (3, 3), 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 1)(x)

                x = Conv2D(
                        filters, (3, 3), 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 2)(x)

                x = AveragePooling2D(pool_size=(2, 2), data_format='channels_first')(x)

                return x

        class CrossStitch(Layer):
                # basic parameter setting
                def __init__(self,input_shape):
                        super(CrossStitch, self).__init__()
                        self.shape = np.prod(input_shape[1:])
                        self.input_shape_1 = self.shape._value
                        self.input_shape_2 = self.shape._value
                        # self.output_shape = [input_shape[1],input_shape[2],input_shape[3]]

                # in cross-stitch network: [xa,xb]*[papameter]=[xa',xb'], the detail refer to the paper
                def build(self, input_shape):
                        
                        shape = self.input_shape_1 + self.input_shape_2
                        self.cross_stitch = self.add_weight(
                                shape=(shape,shape),
                                initializer=tf.initializers.identity(),
                                name='CrossStitch')
                        self.built = True
                        
                # conduct implement of the detailed algorithm calculation
                # inputs represent the output of upper layer, such as x=Dense(parameter)(inputs)
                def call(self,inputs):

                        x1 = Reshape((self.shape,))(inputs[0])
                        x2 = Reshape((self.shape,))(inputs[1])

                        inputss = tf.concat((x1, x2), axis=1)
                        output = tf.matmul(inputss, self.cross_stitch)
                        output1 = output[:,:self.input_shape_1]
                        output2 = output[:,self.input_shape_2:]
                        # print(output1.shape)
                        # print(inputs[0].shape)

                        s1 = inputs[0].shape[1]._value
                        s2 = inputs[0].shape[2]._value
                        s3 = inputs[0].shape[3]._value

                        output1 = tf.reshape(
                                        output1,
                                        shape=[tf.shape(inputs[0])[0],s1,s2,s3])

                        output2 = tf.reshape(
                                        output2,
                                        shape=[tf.shape(inputs[0])[0],s1,s2,s3])

                        return [output1, output2]

        src = Input((1, 64, 64), name = 'src')
        tgt = Input((1, 64, 64), name = 'tgt')

        x1 = src
        x2 = tgt

        num_channels = [ 64, 64, 32, 16, 8, 8]
        if unit_num == 1:
                cs_level = [                 5]
        elif unit_num == 2:
                cs_level = [                4, 5]
        elif unit_num == 3:
                cs_level = [              3,  4, 5]
        elif unit_num == 4:
                cs_level = [              2, 3,  4, 5]
        elif unit_num == 5:
                cs_level = [              1, 2, 3,  4, 5]
        elif unit_num == 6:
                cs_level = [              0, 1, 2, 3,  4, 5]
        for i in range(len(num_channels)):
                # print(num_channels[i])
                x1 = Conv2dBlock(num_channels[i], x1)
                x2 = Conv2dBlock(num_channels[i], x2)

                if i in cs_level:
                        [x1,x2] = CrossStitch(x1.shape)([x1,x2])

        x = Concatenate( axis = 1 )([x1, x2])
        x = Flatten()(x)

        if PASTA == True:

                transformation_params = Dense(7)(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(7):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 2:
                                # y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 3  or i == 4:
                                y_max, y_min = 0.101, -0.101
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i == 5  or i == 6:
                                y_max, y_min = 1.101, 0.901
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_7 = Concatenate( axis = 1, name = 'PASTA_7')(branch_outputs)

                AFF_6 = Lambda(  lambda x:layers.Combining_Affine_Para(x), name = 'AFF_6')([src, PASTA_7])

        else:
                initial_affine_matrix = [
                1.0, 0.0, 0.0, 
                0.0, 1.0, 0.0]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                AFF_6 = Dense(6, bias_initializer=initial_affine_matrix, name = 'AFF_6')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow(x), output_shape = layers.affine_flow_output_shape, name = 'Affine_Warpped')([src, AFF_6])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_6 = AFF_6

        if PASTA == True:
                model.PASTA_7 = PASTA_7

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.NCC_Score, metrics.NCC_Score]
                        )

        return model

def DLIR3D(exp_name, optimizer, PASTA): 

        def Conv3dBlock(filters, x):

                dpr = 0.2
                x = Conv3D(
                        filters, kernel_size=3, 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 1)(x)

                x = Conv3D(
                        filters, kernel_size=3, 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 2)(x)

                x = AveragePooling3D(pool_size=2, data_format='channels_first')(x)

                return x

        src = Input((64, 64, 64), name = 'src')
        tgt = Input((64, 64, 64), name = 'tgt')
        src1 = Lambda(lambda x: K.expand_dims(x, axis=1))(src)
        tgt1 = Lambda(lambda x: K.expand_dims(x, axis=1))(tgt)

        x1 = src1
        x2 = tgt1

        # num_channels = [ 64, 64, 32, 16, 8, 8]
        num_channels = [ 64, 32, 32, 32, 16, 16]

        for level in num_channels:
                # print(level)
                x1 = Conv3dBlock(level, x1)
                x2 = Conv3dBlock(level, x2)

        
        x = Concatenate( axis = 1 )([x1, x2])
        x = Flatten(data_format = 'channels_first')(x)
        if PASTA == True:

                # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
                # Rotation_2D_Range = np.arange(-30,30.01,10)
                # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
                # Scale_2D_Range = np.arange(0.90,1.101,0.05)

                initial_transformation_params = [
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0, 1.0, 1.0]

                initial_transformation_params = tf.constant_initializer(value=initial_transformation_params)

                transformation_params = Dense(15, bias_initializer = initial_transformation_params, name = 'transformation_params')(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(15):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1 or i == 2:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 3 or i == 4 or i == 5:
                                y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                # y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5 and i < 12:
                                y_max, y_min = 0.051, -0.051
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5:
                                y_max, y_min = 1.051, 0.951
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_15 = Concatenate( axis = 1, name = 'PASTA_15')(branch_outputs)

                AFF_12 = Lambda(  lambda x:layers.Combining_Affine_Para3D(x), name = 'AFF_12')([src, PASTA_15])

                

        else:
                initial_affine_matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                ]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                AFF_12 = Dense(12, bias_initializer=initial_affine_matrix, name = 'AFF_12')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow_3D(x), output_shape = layers.affine_flow_3D_output_shape, name = 'Affine_Warpped')([src1, AFF_12])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_12 = AFF_12

        if PASTA == True:
                model.PASTA_15 = PASTA_15

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )
        elif "DCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )
        return model

def CroStchNet3D(exp_name, optimizer,  unit_num, PASTA):
             
        def Conv3dBlock(filters, x):

                dpr = 0.2
                x = Conv3D(
                        filters, kernel_size=3, 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 1)(x)

                x = Conv3D(
                        filters, kernel_size=3, 
                        padding='same', 
                        data_format='channels_first',
                        use_bias=False)(x)

                x = BatchNormalization()(x)
                x = Activation(lambda x : relu(x, alpha = 0.01))(x)
                x = Dropout(dpr, seed = 2)(x)

                x = AveragePooling3D(pool_size=2, data_format='channels_first')(x)

                return x

        class CrossStitch(Layer):
                # basic parameter setting
                def __init__(self,input_shape):
                        super(CrossStitch, self).__init__()
                        self.shape = np.prod(input_shape[1:])
                        self.input_shape_1 = self.shape._value
                        self.input_shape_2 = self.shape._value
                        # self.output_shape = [input_shape[1],input_shape[2],input_shape[3]]

                # in cross-stitch network: [xa,xb]*[papameter]=[xa',xb'], the detail refer to the paper
                def build(self, input_shape):
                        
                        shape = self.input_shape_1 + self.input_shape_2
                        self.cross_stitch = self.add_weight(
                                shape=(shape,shape),
                                initializer=tf.initializers.identity(),
                                name='CrossStitch')
                        self.built = True
                        
                # conduct implement of the detailed algorithm calculation
                # inputs represent the output of upper layer, such as x=Dense(parameter)(inputs)
                def call(self,inputs):

                        x1 = Reshape((self.shape,))(inputs[0])
                        x2 = Reshape((self.shape,))(inputs[1])

                        inputss = tf.concat((x1, x2), axis=1)
                        output = tf.matmul(inputss, self.cross_stitch)
                        output1 = output[:,:self.input_shape_1]
                        output2 = output[:,self.input_shape_2:]
                        # print("output1.shape",output1.shape)
                        # print("inputs[0].shape",inputs[0].shape)

                        s1 = inputs[0].shape[1]._value
                        s2 = inputs[0].shape[2]._value
                        s3 = inputs[0].shape[3]._value
                        s4 = inputs[0].shape[4]._value

                        output1 = tf.reshape(
                                        output1,
                                        shape=[tf.shape(inputs[0])[0],s1,s2,s3,s4])

                        output2 = tf.reshape(
                                        output2,
                                        shape=[tf.shape(inputs[0])[0],s1,s2,s3,s4])
                        # print("output1.shape",output1.shape)
                        return [output1, output2]

        src = Input((64, 64, 64), name = 'src')
        tgt = Input((64, 64, 64), name = 'tgt')
        src1 = Lambda(lambda x: K.expand_dims(x, axis=1))(src)
        tgt1 = Lambda(lambda x: K.expand_dims(x, axis=1))(tgt)

        x1 = src1
        x2 = tgt1

        num_channels = [ 64, 32, 32, 32, 16, 16]
        if unit_num == 1:
                cs_level = [                 5]
        elif unit_num == 2:
                cs_level = [                4, 5]
        elif unit_num == 3:
                cs_level = [              3,  4, 5]
        elif unit_num == 4:
                cs_level = [              2, 3,  4, 5]
        elif unit_num == 5:
                cs_level = [              1, 2, 3,  4, 5]
        elif unit_num == 6:
                cs_level = [              0, 1, 2, 3,  4, 5]
        # cs_level = [                 ]
        for i in range(len(num_channels)):
                # print("cs")
                # print(num_channels[i])
                # print(x1.shape)
                # print(x2.shape)
                x1 = Conv3dBlock(num_channels[i], x1)
                x2 = Conv3dBlock(num_channels[i], x2)
                # print(x1.shape)
                # print(x2.shape)
                if i in cs_level:
                        
                        [x1,x2] = CrossStitch(x1.shape)([x1,x2])

        x = Concatenate( axis = 1 )([x1, x2])
        x = Flatten()(x)

        if PASTA == True:

                # Translation_2D_Range = np.arange(-0.20,0.201,0.1)
                # Rotation_2D_Range = np.arange(-30,30.01,10)
                # Shear_2D_Range = np.arange(-0.1,0.101,0.05)
                # Scale_2D_Range = np.arange(0.90,1.101,0.05)

                initial_transformation_params = [
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        1.0, 1.0, 1.0]

                initial_transformation_params = tf.constant_initializer(value=initial_transformation_params)

                transformation_params = Dense(15, bias_initializer = initial_transformation_params, name = 'transformation_params')(x)

                max_value = 1.0
                Threshold = 0.0

                transformation_params = Activation(lambda x : relu(x, alpha = 0.01, max_value = max_value, threshold = Threshold))(transformation_params)

                branch_outputs = []

                for i in range(15):
                        out = Lambda(lambda x: x[:, i], name = "Splitting_" + str(i))(transformation_params)

                        if i == 0 or i == 1 or i == 2:
                                y_max, y_min = 0.201, -0.201
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)
                        elif i == 3 or i == 4 or i == 5:
                                y_max, y_min = 20.00 * 0.01745, -20.00 * 0.01745
                                # y_max, y_min = 0.34900, -0.34900
                                # y_max, y_min = 30.01 * 0.01745, -30.01 * 0.01745
                                # y_max, y_min = 30.01, -30.01
                                Pasta_Para = Lambda(lambda x: tf.expand_dims(tf.sin((y_max - y_min) * x + y_min),1), name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5 and i < 12:
                                y_max, y_min = 0.051, -0.051
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                        elif i > 5:
                                y_max, y_min = 1.051, 0.951
                                Pasta_Para = Lambda(           lambda x: tf.expand_dims( (y_max - y_min) * x + y_min, 1 ), 
                                                                name = 'Mapping_' + str(i))(out)

                                branch_outputs.append(Pasta_Para)

                PASTA_15 = Concatenate( axis = 1, name = 'PASTA_15')(branch_outputs)

                AFF_12 = Lambda(  lambda x:layers.Combining_Affine_Para3D(x), name = 'AFF_12')([src, PASTA_15])

                

        else:
                initial_affine_matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                ]

                initial_affine_matrix = tf.constant_initializer(value=initial_affine_matrix)

                AFF_12 = Dense(12, bias_initializer=initial_affine_matrix, name = 'AFF_12')(x)


        Affine_Warped = Lambda( lambda x: layers.affine_flow_3D(x), output_shape = layers.affine_flow_3D_output_shape, name = 'Affine_Warpped')([src1, AFF_12])

        model = Model(inputs = [src, tgt], outputs = [Affine_Warped,Affine_Warped])

        model.input_tgt = tgt
        model.output_Affine_Warped = Affine_Warped
        model.AFF_12 = AFF_12

        if PASTA == True:
                model.PASTA_15 = PASTA_15

        if "NCCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "NCCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )
        elif "DCLoss" in exp_name:

                model.compile(
                        optimizer = optimizer,
                        loss = [        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss"),
                                        affine_Loss_zoo(tgt, Affine_Warped, "DCLoss")],
                        loss_weights = [1, 0],
                        metrics=[metrics.DC_Score, metrics.DC_Score]
                        )
        return model