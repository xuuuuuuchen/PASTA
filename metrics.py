
import tensorflow as tf
from keras import backend as K
import numpy as np
EPS = tf.keras.backend.epsilon()

def DC_Score(y_true,y_pred):
    y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1, 64, 64, 64))
    y_pred = tf.reshape(y_pred, (tf.shape(y_true)[0], 1, 64, 64, 64))

    y_true = tf.transpose(y_true, perm=[0, 2, 3, 4, 1])
    y_pred = tf.transpose(y_pred, perm=[0, 2, 3, 4, 1])
    T = y_true
    P = y_pred
    smooth = 1
    return 2*K.sum(T*P)/(K.sum(T) + K.sum(P)+ smooth)

def LNCC_Score(y_true,y_pred):
   
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
    return tf.reduce_mean(cc)

def NCC_Score(y_true, y_pred):

    # print(y_true.shape)
    # print(y_pred.shape)
    
    smooth=tf.keras.backend.epsilon()

    m_y_true, m_y_pred = K.mean(y_true), K.mean(y_pred)
    diff_y_true = (y_true - m_y_true)
    diff_y_pred = (y_pred - m_y_pred)
    nom = K.sum(diff_y_true * diff_y_pred)
    den0 = K.sum(K.square(diff_y_true)) * K.sum(K.square(diff_y_pred))
    den = K.sqrt(den0 + smooth)
    NCC = (nom + smooth) / ( den + smooth)

    return NCC

def DC_2D_Score(y_true,y_pred):
  
    T = y_true
    P = y_pred
    smooth = 1
    return 2*K.sum(T*P)/(K.sum(T) + K.sum(P)+ smooth)
    
def Dice(y_true, y_pred):
    T = (y_true.flatten()>0)
    P = (y_pred.flatten()>0)

    return 2*np.sum(T*P)/(np.sum(T) + np.sum(P))

def NCC(y_true, y_pred):

    smooth=tf.keras.backend.epsilon()

    m_y_true, m_y_pred = np.mean(y_true), np.mean(y_pred)
    diff_y_true = (y_true - m_y_true)
    diff_y_pred = (y_pred - m_y_pred)
    nom = np.sum(diff_y_true * diff_y_pred)
    den0 = np.sum(np.square(diff_y_true)) * np.sum(np.square(diff_y_pred))
    den = np.sqrt(den0 + smooth)
    NCC = (nom + smooth) / ( den + smooth)

    return NCC


def mse(real_list, pred_list):

    error_list = []

    for i in range(len(real_list)):
        error = np.square(np.abs(float(real_list[i]) - pred_list[i]))
        error_list.append(error)

    return error_list

def ncc(y_true, y_pred):

    smooth=tf.keras.backend.epsilon()

    m_y_true, m_y_pred = np.mean(y_true), np.mean(y_pred)
    diff_y_true = (y_true - m_y_true)
    diff_y_pred = (y_pred - m_y_pred)
    nom = np.sum(diff_y_true * diff_y_pred)
    den0 = np.sum(np.square(diff_y_true)) * np.sum(np.square(diff_y_pred))
    den = np.sqrt(den0 + smooth)
    NCC = (nom + smooth) / ( den + smooth)

    return NCC