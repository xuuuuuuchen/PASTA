import tensorflow as tf
import sys
import cv2 # opencv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv
import numpy as np
import csv
from datetime import date

def mgrid(*args, **kwargs):
    """
    create orthogonal grid
    similar to np.mgrid

    Parameters
    ----------
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grid : tf.Tensor [len(args), args[0], ...]
        orthogonal grid
    """
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)


    low = tf.cast(low, tf.float32)
    high = tf.cast(high, tf.float32)

    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> low: "+str(low.eval()))
    #     print(" >>>>>>>>>> high: "+str(high.eval()))
    
    # sess.close()
    coords = (tf.linspace(low, high, arg) for arg in args)
    # print(" >>>>>>>>>> coords: "+str(coords))
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    
    return grid

def batch_mgrid(n_batch, *args, **kwargs):
    """
    create batch of orthogonal grids
    similar to np.mgrid

    Parameters
    ----------
    n_batch : int
        number of grids to create
    args : int
        number of points on each axis
    low : float
        minimum coordinate value
    high : float
        maximum coordinate value

    Returns
    -------
    grids : tf.Tensor [n_batch, len(args), args[0], ...]
        batch of orthogonal grids
    """
    grid = mgrid(*args, **kwargs)
    # with tf.Session() as sess:  
    #     print(" !!!! t: "+str(grid.eval() ) )

    # sess.close()
    # print(" >>>>>>>>>> grid1: "+str(grid.shape))
    grid = tf.expand_dims(grid, 0)
    # print(" >>>>>>>>>> grid2: "+str(grid.shape))
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    # print(" >>>>>>>>>> grid3: "+str(grid.shape))
    return grids

def batch_warp2d(imgs, mappings):
    """
    warp image using mapping function
    I(x) -> I(phi(x))
    phi: mapping function

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    mapping : tf.Tensor
        grids representing mapping function
        [n_batch, xlen, ylen, 2]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """
    n_batch = tf.shape(imgs)[0]
    print(" >>>>>>>>>> mappings: "+str(mappings.shape))
    coords = tf.reshape(mappings, [n_batch, 2, -1])
    print(" >>>>>>>>>> coords: "+str(coords.shape))
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    print(" >>>>>>>>>> x_coords: "+str(x_coords.shape))
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    print(" >>>>>>>>>> y_coords: "+str(y_coords.shape))
    x_coords_flat = tf.reshape(x_coords, [-1])
    print(" >>>>>>>>>> x_coords_flat: "+str(x_coords_flat.shape))
    y_coords_flat = tf.reshape(y_coords, [-1])
    print(" >>>>>>>>>> y_coords_flat: "+str(y_coords_flat.shape))
    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> coords: "+str(coords.eval()))
    #     print(" >>>>>>>>>> x_coords: "+str(x_coords.eval()))
    #     print(" >>>>>>>>>> y_coords: "+str(y_coords.eval()))
    #     print(" >>>>>>>>>> x_coords_flat: "+str(x_coords_flat.eval()))
    #     print(" >>>>>>>>>> y_coords_flat: "+str(y_coords_flat.eval()))

    # sess.close()
    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat)
    print(" >>>>>>>>>> output: "+str(output.shape))
    return output

def offine_warp_for_all(img, array):
    # print(" >>>>>>>>>> output: "+str(output.shape))
    print(" >>>>>>>>>> img: "+str(img.shape))
    print(" >>>>>>>>>> array: "+str(array.shape))
    #  >>>>>>>>>> img: (75, 1, 256, 256)
    # >>>>>>>>>> matrix: (75, 7)
    n_batch = tf.shape(img)[0]
    xlen = tf.shape(img)[2]
    ylen = tf.shape(img)[3]
    n_channel = tf.shape(img)[1]

    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])
   
    """
    CORE
    """
    tx = tf.squeeze(tf.slice(array, [0,0], [n_batch, 1]), 1) 
    ty = tf.squeeze(tf.slice(array, [0,1], [n_batch, 1]), 1)  
    theta = tf.squeeze(tf.slice(array, [0,2], [n_batch, 1]),1)
    cos0 = tf.cos(theta)
    sin0 = tf.sin(theta)
    sx = tf.squeeze(tf.slice(array, [0,3], [n_batch, 1]), 1) 
    sy = tf.squeeze(tf.slice(array, [0,4], [n_batch, 1]), 1)  
    cx = tf.squeeze(tf.slice(array, [0,5], [n_batch, 1]), 1) 
    cy = tf.squeeze(tf.slice(array, [0,6], [n_batch, 1]), 1) 
    x1 = cos0 * cx - sin0 * cx * sx
    x2 = sin0 * cx + cos0 * cx * sx
    x3 = cos0 * cy * sy - sin0 * cy
    x4 = sin0 * cy * sy + cos0 * cy
    x5 = tx
    x6 = ty
    x1 = tf.expand_dims(x1, 1)
    x2 = tf.expand_dims(x2, 1)
    x3 = tf.expand_dims(x3, 1)
    x4 = tf.expand_dims(x4, 1)
    x5 = tf.expand_dims(x5, 1)
    x6 = tf.expand_dims(x6, 1)
    array = tf.concat([x1,x2,x5,x3,x4,x6], 1)

    theta = tf.reshape(array, [-1, 2, 3])

    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    T_g = tf.matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])

    warped_img = batch_warp2d(img, T_g)
    # print(" >>>>>>>>>> warped_img: "+str(warped_img.shape))
    return warped_img


def batch_affine_warp2d(imgs, theta):
    """
    affine transforms 2d images

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    theta : tf.Tensor
        parameters of affine transformation
        [n_batch, 6]

    Returns
    -------
    output : tf.Tensor
        warped images
        [n_batch, xlen, ylen, n_channel]
    """

    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> theta: "+str(theta.shape))

    theta = tf.reshape(theta, [-1, 2, 3])

    # print(" >>>>>>>>>> theta: "+str(theta.shape))
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    # print(" >>>>>>>>>> matrix: "+str(matrix.shape))
    # print(" >>>>>>>>>> matrix: "+str( matrix.eval))
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])
    # print(" >>>>>>>>>> t: "+str(t.shape))
    grids = batch_mgrid(n_batch, xlen, ylen)
    # print(" >>>>>>>>>> grids: "+str(grids.shape))
    coords = tf.reshape(grids, [n_batch, 2, -1])
    # print(" >>>>>            >>>>> coords: "+str(coords.shape))
    # print(" >>>>>            >>>>> matrix: "+str(matrix.shape))
    T_g = tf.matmul(matrix, coords) + t
    # print(" >>>>>>>>>> T_g : "+str(T_g .shape))
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])
    # print(" >>>>>>>>>> T_g : "+str(T_g .shape))
    output = batch_warp2d(imgs, T_g)
    # print(" >>>>>>>>>> output: "+str(output.shape))

    # with tf.Session() as sess:  
    #     print(" >>>>>>sssssssssssss>>>> theta: "+str(theta.eval() ) )
    #     print(" >>>>>>sssssssssssss>>>> matrix: "+str(matrix.eval() ) )
    #     print(" >>>>>>sssssssssssss>>>> t: "+str(t.eval() ) )
    #     print(" >>>>>>sssssssssssss>>>> grids: "+str(grids.eval() ) )
    #     print(" >>>>>>sssssssssssss>>>> coords: "+str(coords.eval() ) )
    #     print(" >>>>>>sssssssssssss>>>> T_g: "+str(T_g.eval() ) )
      
    # sess.close()

    return output, T_g

def _repeat(base_indices, n_repeats):
    # print("base_indices: "+str(base_indices.shape))

    a = tf.reshape(base_indices, [-1, 1])
    b = tf.ones([1, n_repeats], dtype='int32')
    # print("a: "+str(a.shape))
    # print("b: "+str(b.shape))
    base_indices = tf.matmul(a,b)
    # print("base_indices: "+str(base_indices.shape))

    base_indices = tf.reshape(base_indices, [-1])
    # print("base_indices: "+str(base_indices.shape))
    return base_indices

def _interpolate2d(imgs, x, y):
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    n_batch = tf.shape(imgs)[0]
    n_channel = tf.shape(imgs)[1]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    
    # print(" >>>>>>>>>> xlen: "+str(xlen.shape))
    # print(" >>>>>>>>>> ylen: "+str(ylen.shape))
    x = tf.to_float(x)
    y = tf.to_float(y)

    # print(" >>>>>>>>>> x: "+str(x.shape))
    # print(" >>>>>>>>>> y: "+str(y.shape))
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    
    # print(" >>>>>>>>>> xlen: "+str(xlen.shape))
    # print(" >>>>>>>>>> ylen: "+str(ylen.shape))
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    
    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # print(" >>>>>>>>>> x: "+str(x.shape))
    # print(" >>>>>>>>>> y: "+str(y.shape))
    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # print(" >>>>>>>>>> x0: "+str(x0.shape))
    # print(" >>>>>>>>>> y0: "+str(y0.shape))
    # print(" >>>>>>>>>> x1: "+str(x1.shape))
    # print(" >>>>>>>>>> y1: "+str(y1.shape))
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    
    # print(" >>>>>>>>>> x0: "+str(x0.shape))
    # print(" >>>>>>>>>> y0: "+str(y0.shape))
    # print(" >>>>>>>>>> x1: "+str(x1.shape))
    # print(" >>>>>>>>>> y1: "+str(y1.shape))
    base = _repeat(tf.range(n_batch) * xlen * ylen, ylen * xlen)
    # print(" >>>>>>>>>> base: "+str(base.shape))
    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen
    # print(" >>>>>>>>>> base_x0: "+str(base.shape))
    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1
    
    # print(" >>>>>>>>>> index00: "+str(index00.shape))
    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    # print(" >>>>>>>>>> imgs_flat: "+str(imgs_flat.shape))
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)
    


    # print(" >>>>>>>>>> I00: "+str(I00.shape))
    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    cc = I00*w00
    c2 = I01*w01
    c3 = I10*w10
    c4 = I11*w11
    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> xlen: "+str(xlen.eval()))
    #     print(" >>>>>>>>>> ylen: "+str(ylen.eval()))
    #     print(" >>>>>>>>>> x: "+str(x.eval()))
    #     print(" >>>>>>>>>> y: "+str(y.eval()))
    #     print(" >>>>>>>>>> x0: "+str(x0.eval()))
    #     print(" >>>>>>>>>> x1: "+str(x1.eval()))
    #     print(" >>>>>>>>>> y0: "+str(y0.eval()))
    #     print(" >>>>>>>>>> y1: "+str(y1.eval()))
    #     print(" >>>>>>>>>> dx: "+str(dx.eval()))
    #     print(" >>>>>>>>>> dy: "+str(dy.eval()))
    #     print(" >>>>>>>>>> base: "+str(base.eval()))
    #     print(" >>>>>>>>>> base_x0: "+str(base_x0.eval()))
    #     print(" >>>>>>>>>> base_x1: "+str(base_x1.eval()))
    #     print(" >>>>>>>>>> index00 "+str(index00.eval()))
    #     print(" >>>>>>>>>> index01: "+str(index01.eval()))
    #     print(" >>>>>>>>>> index10 "+str(index10.eval()))
    #     print(" >>>>>>>>>> index11: "+str(index11.eval()))
    #     print(" >>>>>>>>>> I00: "+str(I00.eval()))
    #     print(" >>>>>>>>>> I01: "+str(I01.eval()))
    #     print(" >>>>>>>>>> I10: "+str(I10.eval()))
    #     print(" >>>>>>>>>> I11: "+str(I11.eval()))
    #     print(" >>>>>>>>>> dx: "+str(dx.eval()))
    #     print(" >>>>>>>>>> dy: "+str(dy.eval()))
    #     print(" >>>>>>>>>> w00: "+str(w00.eval()))
    #     print(" >>>>>>>>>> w01: "+str(w01.eval()))
    #     print(" >>>>>>>>>> w10: "+str(w10.eval()))
    #     print(" >>>>>>>>>> w11: "+str(w11.eval()))

    #     print(" >>>>>>>>>> I00*w00: "+str(cc.eval()))
    #     print(" >>>>>>>>>> I01*w01: "+str(c2.eval()))
    #     print(" >>>>>>>>>> I10*w10: "+str(c3.eval()))
    #     print(" >>>>>>>>>> I11*w11: "+str(c4.eval()))
    # sess.close()
    # print(" >>>>>>>>>> w11: "+str(w11.shape))
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    # print(" >>>>>>>>>> output: "+str(output.shape))
    # reshape
    output = tf.reshape(output, [n_batch, n_channel, xlen, ylen ])
    # print(" output: "+str(output.shape))


    # delta_x = output[:,1:,:,:] - output[:,:-1,:,:]
    # delta_y = output[:,:,1:,:] - output[:,:,:-1,:]

    # delta_x = delta_x[0, :, :, 0]
    # delta_y = delta_y[0, :, :, 0]

    # print("delta_x " +str(delta_x))
    
    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> delta_x: "+str(delta_x.eval() ) )
    #     print(" >>>>>>>>>> delta_y: "+str(delta_y.eval() ) )
    #     print(" >>>>>>>>>> output: "+str(output.eval() ) )
    #     print(" >>>>>>>>>> d_x: "+str(dx.eval() ) )
    #     print(" >>>>>>>>>> dy: "+str(dy.eval() ) )
    # sess.close()
    # print(" >>>>>>>>>> output: "+str(output ) )
    # print(" >>>>>>>>>> output: "+str(output.shape ) )

    # zeros = tf.zeros([1, 5, 5, 1], tf.float32)

    # indices = tf.constant([[0], [0]])
    # updates = tf.constant([0])
    # update = tf.scatter_nd_update(output[0, :, :, 0], indices, updates)

    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> update: "+str(update.eval() ) )
    # sess.close()

    # output = output[:, 0, 0, :].assign(tf.zeros(1))
    
    # for i in range(imgs.shape[1]-1):
    #     for j in range(imgs.shape[2]-1):
    #         print(i)
    #         # condition = tf.math.equal(delta_x[i,j], 0 )
    #         print("FUCK FUCK FUCK")
    #         output = output[:, i, j, :].assign(tf.zeros(1))
    #         # output =  tf.Variable(output)
    #         # i =  tf.Variable([i])
    #         # j =  tf.Variable([j])
    #         # output = tf.scatter_update(output[0, :, :, 0],[[i, j]],0.0)

    # # print(" output: "+str(output.shape))                      
    # with tf.Session() as sess:  
    #     print(" >>>>>>>>>> output: "+str(output.eval() ) )
    # sess.close()

    return output


   
def save2imgForall(TT, flag, img, flow, foldername,  name1,name2,name3,name4,name5,name6,name7):
	
    resizedimg = tf.image.convert_image_dtype(img, dtype=tf.uint8) 
    resizedimg = tf.squeeze(resizedimg, 0)
    img = tf.image.encode_jpeg(resizedimg)

    resizedflow = tf.image.convert_image_dtype(flow, dtype=tf.uint8) 
    resizedflow = tf.squeeze(resizedflow, 0)
    flow = tf.image.encode_jpeg(resizedflow)

    fname1 = tf.constant('/home/xuchen/Desktop/CVPR20/EXP/DATA/14100-2820/' + foldername + '/tgt/tgt_' + str(flag) + '.jpg')

    # fname2 = tf.constant('/home/xuchen/Desktop/CVPR20/EXP/DATA/14100-2820/' + foldername + '/flow/flow_' + str(flag) + '.jpg')

    fwrite1 = tf.write_file(fname1, img)
    # fwrite2 = tf.write_file(fname2, flow)
    sess = tf.Session()
    result1 = sess.run(fwrite1)
    sess.close()

    # sess = tf.Session()
    # result2 = sess.run(fwrite2)
    # sess.close()

    row = []

    image1name = str(flag)
    image2name = 'tgt_' + str(flag)
    row.append(image1name)
    # row.append(image2name)
    tx = str(round(name1,3))
    ty = str(round(name2,3))
    theta = str(round(name3,3))
    shx = str(round(name4,3))
    shy = str(round(name5,3))
    scx = str(round(name6,3))
    scy = str(round(name7,3))
    row.append(tx)
    row.append(ty)
    row.append(theta)
    row.append(shx)
    row.append(shy)
    row.append(scx)
    row.append(scy)
    row.append(str(foldername))
    # print(row)
    return row

def offline_warpForall(TT, flag, matrix, img, foldername, name1,name2,name3,name4,name5,name6,name7):

	warped_img, flow = batch_affine_warp2d(img, matrix)
	zero_channel = tf.zeros([1, 1, 256, 256])
	flow = tf.concat([flow, zero_channel], axis=1)
	flow = tf.transpose(flow, [0,2,3,1])
	# print(flow.shape)
	row = save2imgForall(TT, flag, warped_img, flow, foldername, name1,name2,name3,name4,name5,name6,name7)

	return row

# def save2img(flag, img, flow, foldername,  name1, name2):
	
#     resizedimg = tf.image.convert_image_dtype(img, dtype=tf.uint8) 
#     resizedimg = tf.squeeze(resizedimg, 0)
#     img = tf.image.encode_jpeg(resizedimg)

#     resizedflow = tf.image.convert_image_dtype(flow, dtype=tf.uint8) 
#     resizedflow = tf.squeeze(resizedflow, 0)
#     flow = tf.image.encode_jpeg(resizedflow)

#     fname1 = tf.constant('/home/xuchen/Desktop/CVPR20/EXP/DATA/14100-2820/' + foldername + '/tgt_' + str(flag) + '.jpg')
#     fname2 = tf.constant('/home/xuchen/Desktop/CVPR20/EXP/DATA/14100-2820/' + foldername + '/flow_' + str(flag) + '='+  str(round(name1,3)) + '+' +str(round(name2,3)) + '.jpg')

#     fwrite1 = tf.write_file(fname1, img)
#     fwrite2 = tf.write_file(fname2, flow)
#     sess = tf.Session()
#     result1 = sess.run(fwrite1)
#     sess.close()

#     sess = tf.Session()
#     result2 = sess.run(fwrite2)
#     sess.close()
#     #######  CVS
#     row = []

#     if foldername == 'Translation':
#         image1name = str(flag)
#         image2name = 'tgt_' + str(flag)
#         row.append(image1name)
#         row.append(image2name)
#         sx = str(0.0)
#         sy = str(0.0)
#         cx = str(1.0)
#         cy = str(1.0)
#         theta = str(0.0)
#         tx = str(round(name1,3))
#         ty = str(round(name2,3))
#         row.append(sx)
#         row.append(sy)
#         row.append(cx)
#         row.append(cy)
#         row.append(theta)
#         row.append(tx)
#         row.append(ty)

#         print(row) 
#         with open('/home/xuchen/Desktop/CVPR20/EXP/DATA/100-40/index.csv', 'a') as csvFile:
#         writer = csv.writer(csvFile)

#     writer.writerow(row)

#     return 

def offline_warp(flag, matrix, img, foldername, name1, name2):

	warped_img, flow = batch_affine_warp2d(img, matrix)
	zero_channel = tf.zeros([1, 1, 256, 256])
	flow = tf.concat([flow, zero_channel], axis=1)
	flow = tf.transpose(flow, [0,2,3,1])
	save2img(flag, warped_img, flow, foldername, name1, name2)

	return 0





    