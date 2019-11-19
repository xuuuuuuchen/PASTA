import tensorflow as tf


def Mapping_Affine_Para(tensors):

    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    
    tx = tf.squeeze(tf.slice(array, [0,0], [n_batch, 1]), 1) 
    print(" >>>>>>>>>> tx: "+str(tx.shape))   
    ty = tf.squeeze(tf.slice(array, [0,1], [n_batch, 1]), 1)  
    theta = tf.squeeze(tf.slice(array, [0,2], [n_batch, 1]),1)
    sx = tf.squeeze(tf.slice(array, [0,3], [n_batch, 1]), 1) 
    sy = tf.squeeze(tf.slice(array, [0,4], [n_batch, 1]), 1)  
    cx = tf.squeeze(tf.slice(array, [0,5], [n_batch, 1]), 1)  
    cy = tf.squeeze(tf.slice(array, [0,6], [n_batch, 1]), 1) 

    tx = tf.expand_dims(0.200 * tx - 0.400, 1)
      
    ty = tf.expand_dims(0.200 * ty - 0.400, 1)

    theta = tf.expand_dims(2.00 * 30.00 * 0.01745 * theta - 30.00 * 0.01745, 1)

    sx = tf.expand_dims(0.200 * sx - 0.400, 1)
    sy = tf.expand_dims(0.200 * sy - 0.400, 1)

    cx = tf.expand_dims(0.100 * cx - 0.800, 1)
    cy = tf.expand_dims(0.100 * cy - 0.800, 1)

    print(" >>>>>>>>>> tx: "+str(tx.shape)) 
    print(" >>>>>>>>>> tx: "+str(ty.shape)) 
    print(" >>>>>>>>>> tx: "+str(theta.shape)) 
    print(" >>>>>>>>>> tx: "+str(sx.shape)) 
    print(" >>>>>>>>>> tx: "+str(sy.shape)) 
    print(" >>>>>>>>>> tx: "+str(cx.shape)) 
    print(" >>>>>>>>>> tx: "+str(cy.shape)) 

    array = tf.concat([tx,ty,theta,sx,sy,cx,cy], 1)
    print(" >>>>>>>>>> array: "+str(array.shape))   
    return array


def Combining_Affine_Para(tensors):

    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    
    tx = tf.squeeze(tf.slice(array, [0,0], [n_batch, 1]), 1) 
    ty = tf.squeeze(tf.slice(array, [0,1], [n_batch, 1]), 1)  
    theta = tf.squeeze(tf.slice(array, [0,2], [n_batch, 1]),1)
    cos0 = tf.cos(theta)
    sin0 = tf.sin(theta)
    sx = tf.squeeze(tf.slice(array, [0,3], [n_batch, 1]), 1) 
    sy = tf.squeeze(tf.slice(array, [0,4], [n_batch, 1]), 1)  
    cx = tf.squeeze(tf.slice(array, [0,5], [n_batch, 1]), 1) 
    cy = tf.squeeze(tf.slice(array, [0,6], [n_batch, 1]), 1) 
    """
    CORE
    """
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
    # print(" >>>>>>>>>> x1: "+str(x1.shape))
    # print(" >>>>>>>>>> x2: "+str(x2.shape))
    # print(" >>>>>>>>>> x3: "+str(x3.shape))
    # print(" >>>>>>>>>> x4: "+str(x4.shape))
    # print(" >>>>>>>>>> x5: "+str(x5.shape))
    # print(" >>>>>>>>>> x6: "+str(x6.shape))
    array = tf.concat([x1,x2,x5,x3,x4,x6], 1)





    # print(" >>>>>>>>>> matrix: "+str(array.shape))   
    return array

def affine_flow(tensors):
    print(" >>>>>>>>>> affine_flow_layer")
    imgs = tensors[0]
    array = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> array: "+str(array.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]


    grids = batch_mgrid(n_batch, xlen, ylen)
    coords = tf.reshape(grids, [n_batch, 2, -1])
   
    theta = tf.reshape(array, [-1, 2, 3])

    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 2])
    t = tf.slice(theta, [0, 0, 2], [-1, -1, -1])

    T_g = tf.matmul(matrix, coords) + t
    T_g = tf.reshape(T_g, [n_batch, 2, xlen, ylen])

    output = batch_warp2d(imgs, T_g)

    return output

def affine_flow_output_shape(input_shapes):

    print(" >>>>>>>>>> input_shapes : "+str(input_shapes))
    shape1 = list(input_shapes[0])
    return (shape1[0],1,shape1[2],shape1[3])


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
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])

    
    return grids

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
    low = tf.to_float(low)
    high = tf.to_float(high)
    coords = (tf.linspace(low, high, arg) for arg in args)
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid

def batch_warp2d(imgs, mappings):
    
    n_batch = tf.shape(imgs)[0]
    coords = tf.reshape(mappings, [n_batch, 2, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # imgs = tf.transpose(imgs,[0,2,3,1])
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat)
    return output

def _interpolate2d(imgs, x, y):
    print("interpolate2d")

    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[1]

    x = tf.to_float(x)
    y = tf.to_float(y)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    def _repeat(base_indices, n_repeats):
        base_indices = tf.matmul(
            tf.reshape(base_indices, [-1, 1]),
            tf.ones([1, n_repeats], dtype='int32'))
        return tf.reshape(base_indices, [-1])

    base = _repeat(tf.range(n_batch) * xlen * ylen, ylen * xlen)

    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen

    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    # reshape
    output = tf.reshape(output, [n_batch, n_channel, xlen, ylen])

    return output




def Split_sx(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    sx = tf.slice(array, [0,0], [n_batch, 1])
    return sx

def Split_sy(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    sy = tf.slice(array, [0,1], [n_batch, 1])
    return sy

def Split_cx(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    cx = tf.slice(array, [0,2], [n_batch, 1])
    return cx

def Split_cy(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    cy = tf.slice(array, [0,3], [n_batch, 1])
    return cy

def Split_theta(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    theta = tf.slice(array, [0,4], [n_batch, 1]) 
    return theta

def Split_tx(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    tx = tf.slice(array, [0,5], [n_batch, 1])
    return tx

def Split_ty(tensors):
    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    ty = tf.slice(array, [0,6], [n_batch, 1])
    return ty

"""
x1 = cos0 * cx - sin0 * cx * sx
x2 = sin0 * cy + cos0 * cy * sx
x3 = cos0 * cx * sy - sin0 * cx
x4 = sin0 * cy * sy + cos0 * cy
x5 = tx
x6 = ty
"""
def combine_x1(tensors):
    theta = tensors[0]
    cx = tensors[1]
    sx = tensors[2]
    x1 = tf.cos(theta) * cx - tf.sin(theta) * cx * sx
    return x1

def combine_x2(tensors):
    theta = tensors[0]
    cy = tensors[1]
    sx = tensors[2]
    x2 = tf.sin(theta) * cy + tf.cos(theta) * cy * sx
    return x2
def combine_x3(tensors):
    theta = tensors[0]
    cx = tensors[1]
    sy = tensors[2]
    x3 = tf.cos(theta) * cx * sy - tf.sin(theta) * cx
    return x3
def combine_x4(tensors):
    theta = tensors[0]
    cy = tensors[1]
    sy = tensors[2]
    x4 = tf.sin(theta) * cy * sy + tf.cos(theta) * cy
    return x4


def Mapping_Squeezed_Affine_Para(tensors):

        print(" >>>>>>>>>> Mapping_7_Affine_Para_layer")
        imgs = tensors[0]
        Squeezed_Affine_Para = tensors[1]
        print(" >>>>>>>>>> imgs: "+str(imgs.shape))
        print(" >>>>>>>>>> Squeezed_Affine_Para: "+str(Squeezed_Affine_Para.shape))
        n_batch = tf.shape(imgs)[0]
        

        sx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,0], [n_batch, 1]), 1) 
        sy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,1], [n_batch, 1]), 1) 
        cx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,2], [n_batch, 1]), 1) 
        cy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,3], [n_batch, 1]), 1) 
        theta = tf.slice(Squeezed_Affine_Para, [0,4], [n_batch, 1]) 
        theta = tf.squeeze(theta, 1)
        print(" >>>>>>>>>> theta: "+str(theta.shape))
        cos0 = tf.cos(theta)
        sin0 = tf.sin(theta)
        tx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,5], [n_batch, 1]), 1) 
        ty = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,6], [n_batch, 1]), 1) 


        # sx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,0], [n_batch, 1]), 1) * 0.000001
        # sy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,1], [n_batch, 1]), 1) * 0.000001
        # cx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,2], [n_batch, 1]), 1) * 0.000001 + 1.0
        # cy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,3], [n_batch, 1]), 1) * 0.000001 + 1.0
        # theta = tf.slice(Squeezed_Affine_Para, [0,4], [n_batch, 1]) 

        # theta = tf.squeeze(theta, 1) * 0.4 - 0.2
        # print(" >>>>>>>>>> theta: "+str(theta.shape))
        # cos0 = tf.cos(theta)
        # sin0 = tf.sin(theta)

        # tx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,5], [n_batch, 1]), 1) * 0.000001
        # ty = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,6], [n_batch, 1]), 1) * 0.000001

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

        Affine_Para_7 = 2 * Affine_Para_7 - 1
        print(" >>>>>>>>>> Affine_Para_7: "+str(Affine_Para_7.shape))

        return Affine_Para_7


def non_linear_warp_2d(tensors):
    print(" non_linear_warp_2d")
    imgs = tensors[0]
    vector_fields = tensors[1]
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]

    grids = non_linear_mgrid(n_batch, xlen, ylen)
    print(" >>>>>>>>>> grids: "+str(grids.shape))
    
    T_g = grids + vector_fields
    print(" >>>>>>>>>> T_g: "+str(T_g.shape))
    
    output = non_linear_warp2d(imgs, T_g)
    return output

def non_linear_mgrid(n_batch, *args, **kwargs):
    
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])

    return grids

def non_linear_warp2d(imgs, mappings):
    
    n_batch = tf.shape(imgs)[0]
    coords = tf.reshape(mappings, [n_batch, 2, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    print("non_linear_warp2d")
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # imgs = tf.transpose(imgs,[0,2,3,1])
    print(" >>>>>>>>>> imgs: "+str(imgs.shape))

    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat)
    return output

def _repeat(base_indices, n_repeats):
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    return tf.reshape(base_indices, [-1])

def _interpolate2d(imgs, x, y):
    print("interpolate2d")

    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    n_channel = tf.shape(imgs)[1]

    x = tf.to_float(x)
    y = tf.to_float(y)
    xlen_f = tf.to_float(xlen)
    ylen_f = tf.to_float(ylen)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    base = _repeat(tf.range(n_batch) * xlen * ylen, ylen * xlen)

    base_x0 = base + x0 * ylen
    base_x1 = base + x1 * ylen

    index00 = base_x0 + y0
    index01 = base_x0 + y1
    index10 = base_x1 + y0
    index11 = base_x1 + y1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.to_float(imgs_flat)
    I00 = tf.gather(imgs_flat, index00)
    I01 = tf.gather(imgs_flat, index01)
    I10 = tf.gather(imgs_flat, index10)
    I11 = tf.gather(imgs_flat, index11)

    # and finally calculate interpolated values
    dx = x - tf.to_float(x0)
    dy = y - tf.to_float(y0)
    w00 = tf.expand_dims((1. - dx) * (1. - dy), 1)
    w01 = tf.expand_dims((1. - dx) * dy, 1)
    w10 = tf.expand_dims(dx * (1. - dy), 1)
    w11 = tf.expand_dims(dx * dy, 1)
    output = tf.add_n([w00*I00, w01*I01, w10*I10, w11*I11])
    # reshape
    output = tf.reshape(output, [n_batch, n_channel, xlen, ylen])

    return output
