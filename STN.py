import tensorflow as tf

import sys
sys.path.insert(0, '/home/xuchen/Desktop/CVPR20/EXP/LIBS')
import tensorflow as tf
from grid import batch_mgrid
from warp import batch_warp2d
from warp_tf import batch_warp2d_tf
import numpy as np
from keras import backend as K
from skimage.transform import warp, AffineTransform
from skimage.io import imread
from skimage import io; io.use_plugin('matplotlib')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import ProjectiveTransform



def warp_2d_layer(tensors):
    print(" >>>>>>>>>> warp_2d_layer")
   
    imgs = tensors[0]
    vector_fields = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))

    # imgs = K.permute_dimensions(imgs, [0, 2, 3, 1])

    vector_fields = K.permute_dimensions(vector_fields, [0, 2, 3, 1])

    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))

    n_batch = K.shape(imgs)[0]
    xlen = K.shape(imgs)[1]
    ylen = K.shape(imgs)[2]

    grids = batch_mgrid(n_batch, xlen, ylen)
    grids = K.permute_dimensions(grids, [0, 2, 3, 1])
    print(" >>>>>>>>>> grids: "+str(grids.shape))

    T_g = grids + vector_fields
    print(" >>>>>>>>>> T_g: "+str(T_g.shape))

    output = batch_warp2d(imgs, T_g)
    print(" >>>>>>>>>> output: "+str(output.shape))
    print(" >>>>>>>>>> output: "+str(output))
    return output


def batch_displacement_warp2d(tensors):
    print(" batch_displacement_warp2d")
    imgs = tensors[0]
    vector_fields = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]

    grids = batch_mgrid(n_batch, xlen, ylen)
    print(" >>>>>>>>>> grids: "+str(grids.shape))
    
    # T_g = grids + vector_fields
    # print(" >>>>>>>>>> T_g: "+str(T_g.shape))
    
    output = batch_warp2d(imgs, vector_fields)
    return output

def batch_displacement_warp2d_tf(tensors):

    imgs = np.moveaxis(imgs,1,-1)
    # vector_fields = np.moveaxis(vector_fields,1,-1)
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))
    grids = batch_mgrid(n_batch, xlen, ylen)
    print(" >>>>>>>>>> grids: "+str(grids.shape))

    # T_g = grids + vector_fields
    # print(" >>>>>>>>>> T_g: "+str(T_g.shape))

    output = batch_warp2d_tf(imgs, vector_fields)
    print(" >>>>>>>>>> output: "+str(output.shape))
    print(" >>>>>>>>>> output: "+str(output))
    return output


def warp_2d_layer_output_shape(input_shapes):
  
    shape1 = list(input_shapes[0])
    # print(" >>>>>>>>>> shape1: "+str(tuple(shape1).shape))
    return tuple(shape1)



###################################################
# AFFINE
###################################################

def affineT_offline_layer(tensors):

    print(" affineT_offline_layer")
    imgs = tensors[0]
    perd_Affine_Para = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> perd_Affine_Para: "+str(perd_Affine_Para.shape))

    n_batch = tf.shape(imgs)[0]
    all_warped_images = []
      
    for i in range(952):
        print(i)
        tf_matrix = perd_Affine_Para[i,:,:]
        print(" >>>>>>>>>> tf_matrix: "+str(tf_matrix.shape))
        image = imgs[i,:,:,:]
        # print("image.sahpe" + str(image.shape))

        newrow = [0,0,1]
        newrow = np.array(newrow)
        # print(" >>>>>>>>>> newrow: "+str(newrow.shape))
        newrow = np.reshape(newrow, (1, 3))
        print(" >>>>>>>>>> newrow: "+str(newrow.shape))

        tf_matrix = np.vstack((tf_matrix, newrow))
        print(" >>>>>>>>>> tf_matrix: "+str(tf_matrix.shape))

        # aff_a0 = np.asscalar(np.array(tf_matrix[0,0]))
        # aff_a1 = np.asscalar(np.array(tf_matrix[1]))
        # aff_a2 = np.asscalar(np.array(tf_matrix[2]))
        # aff_b0 = np.asscalar(np.array(tf_matrix[3]))
        # aff_b1 = np.asscalar(np.array(tf_matrix[4]))
        # aff_b2 = np.asscalar(np.array(tf_matrix[5]))

        aa = K.constant([1])
        print(type(tf.Session().run(aa)))
        print(aa)
        aff_aa = K.eval(aa)
        print(aff_aa)

        
        aat= aff_a0
        aat = K.constant(aat[0])
        print(type(tf.Session().run(aat)))


        aatt = K.constant([aat])

        
        # tt_s = tf.Variable(aat, name='tts')
        tt_s = K.constant(aat)
        tt_s = K.eval(tt_s)

        print(tt_s)
        bb = tf.constant(tt_s)
        print(type(tf.Session().run(bb)))
        print(bb)
        aff_bb = K.eval(bb)
        print(aff_bb)



    
        tf_matrix2 = [0,1]


        x0 = np.asscalar(np.array(tf_matrix2[0]))
        y0 = x0
        z0 = np.asscalar(np.array(tf_matrix2[1]))

        x = tf.cast(x0, "float32")
        y = tf.cast(y0, "float32")
        z = tf.cast(z0, "float32")


        matrix = [          [aff_a0, aff_a1, aff_a2],
                            [aff_b0, aff_b1, aff_b2],
                            [x, y, z]]
        
        # print(matrix.shape)
        print(image.shape)
        
        image_s = tf.squeeze(image, axis=0)
        # newimage = image[]
    
 
        
        image_s = tf.cast(image_s, "float64")

        print("warping!!!")
        print(matrix)
        print(image)
        print(image_s)
        # sess = tf.Session()
        # sess.run(tf.SparseTensor(image_s))
        # tf.constant(image_s).eval()



        # matrix = [ [0, 0, 0],
        #                     [0, 0, 0],
        #                     [0, 0, 0]]


        # matrix - np.asarray(matrix)

        warped_image = warp(image_s, matrix)
        # matrix - np.array(matrix)
        # matrix = tf.cast(matrix, "float64")
        # warped_image = warp(matrix, ProjectiveTransform(matrix=matrix))

        # print("warped_image.sahpe" + str(warped_image.shape))

        all_warped_images.append(warped_image)

    all_warped_images = np.array(all_warped_images)
 
    return all_warped_images

def affineT_offline_2_layer(tensors):

    imgs = tensors[0]
    perd_Affine_Para = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> perd_Affine_Para: "+str(perd_Affine_Para.shape))

    n_batch = tf.shape(imgs)[0]
    all_matrix = []
      
    for i in range(952):
        # print(i)
        tf_matrix = perd_Affine_Para[i,:,:,:]
        # print("tf_matrix.sahpe" + str(tf_matrix.shape))
         
        aff_a0_T = tf_matrix[0]


        aff_a0 = np.asscalar(np.array(tf_matrix[0]))
        aff_a1 = np.asscalar(np.array(tf_matrix[1]))
        aff_a2 = np.asscalar(np.array(tf_matrix[2]))
        aff_b0 = np.asscalar(np.array(tf_matrix[3]))
        aff_b1 = np.asscalar(np.array(tf_matrix[4]))
        aff_b2 = np.asscalar(np.array(tf_matrix[5]))


        tf_matrix2 = [0,1]


        x0 = np.asscalar(np.array(tf_matrix2[0]))
        y0 = x0
        z0 = np.asscalar(np.array(tf_matrix2[1]))

        x = tf.cast(x0, "float32")
        y = tf.cast(y0, "float32")
        z = tf.cast(z0, "float32")


        matrix = [ [aff_a0, aff_a1, aff_a2],
                            [aff_b0, aff_b1, aff_b2],
                            [x, y, z]]
        
        

        all_matrix.append(matrix)

    all_matrix = np.array(all_matrix)
 
    return all_matrix



def affineT_offline_layer_output(input_shapes):
  
    shape1 = list(input_shapes[0])

    return tuple(shape1)

def affineT_offline_2_layer_output(input_shapes):
  
    shape1 = list(input_shapes[0])

    return tuple(shape1)


def Mapping_Squeezed_Affine_Para(tensors):

    print(" >>>>>>>>>> Mapping_7_Affine_Para_layer")
    imgs = tensors[0]
    Squeezed_Affine_Para = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> Squeezed_Affine_Para: "+str(Squeezed_Affine_Para.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])
    print(" >>>>>>>>>> grids: "+str(grids.shape))
    print(" >>>>>>>>>> coords: "+str(coords.shape))

    sx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,0], [n_batch, 1]), 1) * 0.0
    sy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,1], [n_batch, 1]), 1) * 0.0
    cx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,2], [n_batch, 1]), 1) * 0.0 + 1.0
    cy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,3], [n_batch, 1]), 1) * 0.0 + 1.0
    theta = tf.slice(Squeezed_Affine_Para, [0,4], [n_batch, 1]) *100
    cos0 = tf.squeeze(tf.cos(theta), 1)
    sin0 = tf.squeeze(tf.sin(theta), 1)
    theta = tf.squeeze(theta, 1)
    tx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,5], [n_batch, 1]), 1) * 0.0
    ty = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,6], [n_batch, 1]), 1) * 0.0

    print(" >>>>>>>>>> 1.sx: "+str(sx.shape))
    print(" >>>>>>>>>> 2.sy: "+str(sy.shape))
    print(" >>>>>>>>>> 3.cx: "+str(sx.shape))
    print(" >>>>>>>>>> 4.cy: "+str(sy.shape))
    print(" >>>>>>>>>> 5.theta: "+str(theta.shape))
    print(" >>>>>>>>>> 5. cos0: "+str(cos0.shape))
    print(" >>>>>>>>>> 5. sin0: "+str(sin0.shape))
    print(" >>>>>>>>>> 6.tx: "+str(tx.shape))
    print(" >>>>>>>>>> 7.ty: "+str(ty.shape))

    sx = tf.expand_dims(sx, 1)
    sy = tf.expand_dims(sy, 1)
    cx = tf.expand_dims(cx, 1)
    cy = tf.expand_dims(cy, 1)
    theta = tf.expand_dims(theta, 1)
    tx = tf.expand_dims(tx, 1)
    ty = tf.expand_dims(ty, 1)

    Affine_Para_7 = tf.concat([sx, sy, cx, cy, theta, tx, ty], 1)
    Affine_Para_7 = tf.expand_dims(Affine_Para_7, 2)
    Affine_Para_7 = tf.squeeze(Affine_Para_7, 2)
    print(" >>>>>>>>>> Affine_Para_7: "+str(Affine_Para_7.shape))

    return Affine_Para_7



def affine_flow(tensors):

    print(" >>>>>>>>>> affine_flow_layer")
    imgs = tensors[0]
    Squeezed_Affine_Para = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> Squeezed_Affine_Para: "+str(Squeezed_Affine_Para.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])
    print(" >>>>>>>>>> grids: "+str(grids.shape))
    print(" >>>>>>>>>> coords: "+str(coords.shape))

    sx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,0], [n_batch, 1]), 1) 
    sy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,1], [n_batch, 1]), 1)  
    cx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,2], [n_batch, 1]), 1)  
    cy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,3], [n_batch, 1]), 1) 
    theta = tf.slice(Squeezed_Affine_Para, [0,4], [n_batch, 1]) 
    cos0 = tf.squeeze(tf.cos(theta), 1)
    sin0 = tf.squeeze(tf.sin(theta), 1)
    theta = tf.squeeze(theta, 1)
    tx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,5], [n_batch, 1]), 1) 
    ty = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,6], [n_batch, 1]), 1)  

    x1 = cos0 * cx - sin0 * cx * sx
    x2 = sin0 * cy + cos0 * cy * sx
    x3 = cos0 * cx * sy - sin0 * cx
    x4 = sin0 * cy * sy + cos0 * cy
    x5 = tx
    x6 = ty
    
    x1 = tf.expand_dims(x1, 1)
    x2 = tf.expand_dims(x2, 1)
    x3 = tf.expand_dims(x3, 1)
    x4 = tf.expand_dims(x4, 1)
    x5 = tf.expand_dims(x5, 1)
    x6 = tf.expand_dims(x6, 1)

    print(" >>>>>>>>>> x1: "+str(x1.shape))
    print(" >>>>>>>>>> x2: "+str(x2.shape))
    print(" >>>>>>>>>> x3: "+str(x3.shape))
    print(" >>>>>>>>>> x4: "+str(x4.shape))
    print(" >>>>>>>>>> x5: "+str(x5.shape))
    print(" >>>>>>>>>> x6: "+str(x6.shape))

    x1x2 = tf.concat([x1,x2], 1)
    x1x2 = tf.expand_dims(x1x2, 2)
    x3x4 = tf.concat([x3,x4], 1)
    x3x4 = tf.expand_dims(x3x4, 2)
    x5x6 = tf.concat([x5,x6], 1)
    x5x6 = tf.expand_dims(x5x6, 2)

    print(" >>>>>>>>>> x1x2: "+str(x1x2.shape))
    print(" >>>>>>>>>> x3x4: "+str(x3x4.shape))
    print(" >>>>>>>>>> x5x6: "+str(x5x6.shape))

    x1x2x3x4 = tf.concat([x1x2,x3x4], 2)
    print(" >>>>>>>>>> x1x2x3x4 "+str(x1x2x3x4.shape))

    matrix = x1x2x3x4
    print(" >>>>>>>>>> matrix "+str(matrix.shape))
    t = x5x6
    print(" >>>>>>>>>> t: "+str(t.shape))

    T_g = tf.matmul(matrix, coords) + t
    print(" >>>>>>>>>> T_g : "+str(T_g .shape))
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    print(" >>>>>>>>>> T_g : "+str(T_g .shape))
    return T_g


def affine_flow_output_shape(input_shapes):
    print(" >>>>>>>>>> input_shapes : "+str(input_shapes))
    shape1 = list(input_shapes[0])
    print(" >>>>>>>>>> shape1 : "+str(shape1))
    # return tuple(shape1)
    return (shape1[0],2,shape1[2],shape1[3])



def Mapping_each_aff_para(tensor):

    print(" >>>>>>>>>> Mapping_each_aff_para_layer")
    print(" >>>>>>>>>> tensor: "+str(tensor.shape))
    Affine_Para = tensor
    Affine_Para = 2 * Affine_Para - 10

    return Affine_Para 













