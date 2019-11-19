import tensorflow as tf
import keras.backend as K

# def NMI_Score(y_true, y_pred):
#     return NMI_Loss(y_true, y_pred)

def NCC_Score(y_true, y_pred):

    m_y_true, m_y_pred = K.mean(y_true), K.mean(y_pred)

    # std_y_true = K.std(y_true, axis = 1, keepdims=True)
    # std_y_pred = K.std(y_pred, axis = 1, keepdims=True)

    diff_y_true = (y_true - m_y_true)
    diff_y_pred = (y_pred - m_y_pred)

    # diff_y_true = (y_true - m_y_true)/(std_y_true + 0.000001)
    # diff_y_pred = (y_pred - m_y_pred)/(std_y_pred + 0.000001)

    num = K.sum(diff_y_true * diff_y_pred)

    den0 = K.sum(K.square(diff_y_true)) * K.sum(K.square(diff_y_pred))

    den = K.sqrt(den0 + 0.000001)

    NCC = num / ( den + 0.000001)

    return NCC

# def Huber_Score(y_true, y_pred):
#     return Huber_Loss(y_true, y_pred)

# def SSD_Score(y_true, y_pred):
#     return SSD_Loss(y_true, y_pred)

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
            ssd_loss = SSD_Loss(y_true, y_pred)
            return ssd_loss

    return loss

def Flow_Smooth_Loss(y):
    delta_y = y[:,:,1:,:] - y[:,:,:-1,:]
    smooth_loss = K.sum(K.abs(delta_y[:,:,1:,:-2])**2) 
    return smooth_loss

def my_flow_smooth_loss(y_pred):

    print ("my_flow_loss")
    y_pred = y_pred[:,0:2,:,:]
    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,1:,:-2]**2
    delta_y = y[:,:,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y) 
    epsilon = 0.00000001
    delta = K.sum(K.sqrt(delta_u + epsilon))
    return delta

def Non_linear_Loss(tgt, Non_linear_Warped, flow, loss_name):

    y_true = tgt
    y_pred = Non_linear_Warped

    w = 1
    smooth_loss = my_flow_smooth_loss(flow)

    if loss_name == "NCC_Loss":
        
        def loss(y_true, y_pred):
            ncc_loss = NCC_Loss(y_true, y_pred) + w * smooth_loss/(256*256)
            return ncc_loss

    if loss_name == "NMI_Loss":

        def loss(y_true, y_pred):
            nmi_loss = NMI_Loss(y_true, y_pred) + w * smooth_loss /(256*256)
            return nmi_loss

    if loss_name == "Huber_Loss":
        def loss(y_true, y_pred):
            Huber_loss = Huber_Loss(y_true, y_pred) + w * smooth_loss /(256*256)
            return Huber_loss

    if loss_name == "SSD_Loss":

        def loss(y_true, y_pred):
            ssd_loss = SSD_Loss(y_true, y_pred) + w * smooth_loss /(256*256)
            return ssd_loss
            

    return loss 



