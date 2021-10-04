

import tensorflow as tf
from keras import backend as K
import metrics
import numpy as np
EPS = tf.keras.backend.epsilon()

def NCC_loss(y_true, y_pred):
    return  1 - metrics.NCC_Score(y_true, y_pred)


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

def SSD_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred) )

    
    # y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1, 64, 64, 64))
    # y_pred = tf.reshape(y_pred, (tf.shape(y_true)[0], 1, 64, 64, 64))

    # y_true = tf.transpose(y_true, perm=[0, 2, 3, 4, 1])
    # y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])
    # print("!!!!!!!!!!! y_true.shape ", y_true.shape)
    # print("!!!!!!!!!!! y_pred.shape ", y_pred.shape)

    # I = y_true
    # J = y_pred
    # I2 = I*I
    # J2 = J*J
    # IJ = I*J
    # win=[9,9,9]
    # filt = tf.ones([win[0], win[1], win[2], 1, 1]) 

    # I_sum = tf.nn.conv3d(I, filt, [1, 1, 1, 1, 1], "SAME")
    # J_sum = tf.nn.conv3d(J, filt, [1, 1, 1, 1, 1], "SAME")
    # I2_sum = tf.nn.conv3d(I2, filt, [1, 1, 1, 1, 1], "SAME")
    # J2_sum = tf.nn.conv3d(J2, filt, [1, 1, 1, 1, 1], "SAME")
    # IJ_sum = tf.nn.conv3d(IJ, filt, [1, 1, 1, 1, 1], "SAME")

    # win_size = win[0]*win[1]*win[2]
    # u_I = I_sum/win_size
    # u_J = J_sum/win_size

    # cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    # I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    # J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    # cc = cross*cross / (I_var*J_var+1e-5)

    # # if(voxel_weights is not None):
    # #	cc = cc * voxel_weights

    # #return -tf.log(tf.reduce_mean(cc))
    # # return 1/tf.reduce_mean(cc)-1
    # return 1-tf.reduce_mean(cc)
    # # return (1-tf.reduce_mean(cc))**2
    # # return 1/(tf.reduce_mean(cc)+1+1e-5)-0.5

def LNCC_loss(y_true,y_pred):
    
    
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1, 64, 64, 64))
    y_pred = tf.reshape(y_pred, (tf.shape(y_true)[0], 1, 64, 64, 64))

    y_true = tf.transpose(y_true, perm=[0, 2, 3, 4, 1])
    y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])
    print("!!!!!!!!!!! y_true.shape ", y_true.shape)
    print("!!!!!!!!!!! y_pred.shape ", y_pred.shape)

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    I = y_true
    J = y_pred
    ndims = len(I.get_shape().as_list()) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    win = [9] * ndims

    # get convolution function
    conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I*I
    J2 = J*J
    IJ = I*J

    # compute filters
    sum_filt = tf.ones([9, 9, 9, 1, 1])
    strides = 1
    if ndims > 1:
        strides = [1] * (ndims + 2)
    padding = 'SAME'

    # compute local sums via convolution
    I_sum = conv_fn(I, sum_filt, strides, padding)
    J_sum = conv_fn(J, sum_filt, strides, padding)
    I2_sum = conv_fn(I2, sum_filt, strides, padding)
    J2_sum = conv_fn(J2, sum_filt, strides, padding)
    IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

    # compute cross correlation
    win_size = np.prod(win)
    u_I = I_sum/win_size
    u_J = J_sum/win_size

    cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

    cc = cross*cross / (I_var*J_var + EPS)

    # return negative cc.
    return - tf.reduce_mean(cc)

def DC_loss(y_true,y_pred):

    # def Dice(y_true, y_pred):
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1, 64, 64, 64))
    y_pred = tf.reshape(y_pred, (tf.shape(y_true)[0], 1, 64, 64, 64))

    y_true = tf.transpose(y_true, perm=[0, 2, 3, 4, 1])
    y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])
    # print("!!!!!!!!!!! y_true.shape ", y_true.shape)
    # print("!!!!!!!!!!! y_pred.shape ", y_pred.shape)
    T = y_true
    P = y_pred

    smooth = 1
    return 1-2*K.sum(T*P)/(K.sum(T) + K.sum(P)+ smooth)


def DC_2D_loss(y_true,y_pred):

    # def Dice(y_true, y_pred):
    # y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1, 64, 64, 64))
    # y_pred = tf.reshape(y_pred, (tf.shape(y_true)[0], 1, 64, 64, 64))

    # y_true = tf.transpose(y_true, perm=[0, 2, 3, 4, 1])
    # y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])
    # print("!!!!!!!!!!! y_true.shape ", y_true.shape)
    # print("!!!!!!!!!!! y_pred.shape ", y_pred.shape)
    T = y_true
    P = y_pred

    smooth = 1
    return 1-2*K.sum(T*P)/(K.sum(T) + K.sum(P)+ smooth)
    

def NMI_Loss(y_true, y_pred):
               
    smooth=tf.keras.backend.epsilon()
    y_true = K.clip(y_true, smooth, 1)
    y_pred = K.clip(y_pred, smooth, 1)
    KL_loss = K.sum(y_true * K.log(y_true / y_pred), axis = 1) 
    entropy_y_true = K.sum(y_true * K.log(y_true))
    entropy_y_pred = K.sum(y_pred * K.log(y_pred))
    NMI_loss = (2 * KL_loss + smooth) / (entropy_y_true * entropy_y_pred + smooth)
    return NMI_loss

def SSD_Loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred) )