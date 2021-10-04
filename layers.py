import tensorflow as tf



def Combining_Affine_Para(tensors):

    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    
    tx = tf.squeeze(tf.slice(array, [0,0], [n_batch, 1]), 1) 
    ty = tf.squeeze(tf.slice(array, [0,1], [n_batch, 1]), 1)  

    sin0 = tf.squeeze(tf.slice(array, [0,2], [n_batch, 1]),1)
    cos0 = tf.sqrt(1.0-tf.square(sin0))

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

def Combining_Affine_Para3D(tensors):

    imgs = tensors[0]
    array = tensors[1]
    n_batch = tf.shape(imgs)[0]
    
    tx = tf.squeeze(tf.slice(array, [0,0], [n_batch, 1]), 1) 
    ty = tf.squeeze(tf.slice(array, [0,1], [n_batch, 1]), 1)  
    tz = tf.squeeze(tf.slice(array, [0,2], [n_batch, 1]), 1)  

    sin0x = tf.squeeze(tf.slice(array, [0,3], [n_batch, 1]),1)
    sin0y = tf.squeeze(tf.slice(array, [0,4], [n_batch, 1]),1)
    sin0z = tf.squeeze(tf.slice(array, [0,5], [n_batch, 1]),1)
    cos0x = tf.sqrt(1.0-tf.square(sin0x))
    cos0y = tf.sqrt(1.0-tf.square(sin0y))
    cos0z = tf.sqrt(1.0-tf.square(sin0z))

    shxy = tf.squeeze(tf.slice(array, [0,6], [n_batch, 1]), 1) 
    shyx = tf.squeeze(tf.slice(array, [0,7], [n_batch, 1]), 1) 
    shxz = tf.squeeze(tf.slice(array, [0,8], [n_batch, 1]), 1) 
    shzx = tf.squeeze(tf.slice(array, [0,9], [n_batch, 1]), 1) 
    shzy = tf.squeeze(tf.slice(array, [0,10], [n_batch, 1]), 1) 
    shyz = tf.squeeze(tf.slice(array, [0,11], [n_batch, 1]), 1) 

    scx = tf.squeeze(tf.slice(array, [0,12], [n_batch, 1]), 1) 
    scy = tf.squeeze(tf.slice(array, [0,13], [n_batch, 1]), 1) 
    scz = tf.squeeze(tf.slice(array, [0,14], [n_batch, 1]), 1) 
    """
    CORE
    """
    x1 = -shxz*scx*sin0y+scx*cos0y*cos0z+shxy*scx*cos0y*sin0z
    
    x2 = shxz*scx*sin0x*cos0y+shxy*scx*cos0x*cos0z +scx*sin0x*sin0y*cos0z-scx*cos0x*sin0z+shxy*scx*sin0x*sin0y*sin0z

    x3 = shxz*scx*cos0x*cos0y-shxy*scx*sin0x*cos0z+scx*cos0x*sin0y*cos0z+scx*sin0x*sin0z+shxy*scx*cos0x*sin0y*sin0z

    x4 = scy*cos0y*sin0z+scy*cos0y*cos0z*shyx-scy*sin0y*shyz

    x5 = scy*cos0x*cos0z+scy*sin0x*sin0y*sin0z+scy*sin0x*sin0y*cos0z*shyx-scy*cos0x*sin0z*shyx+scy*sin0x*cos0y*shyz

    x6 = -scy*sin0x*cos0z+scy*cos0x*sin0y*sin0z+scy*cos0x*sin0y*cos0z*shyx+scy*sin0x*sin0z*shyx+scy*cos0x*cos0y*shyz

    x7 = -scz*sin0y+shzx*scz*cos0y*cos0z+shzy*scz*cos0y*sin0z

    x8 = scz*sin0x*cos0y+shzy*scz*cos0x*cos0z+shzx*scz*sin0x*sin0y*cos0z-shzx*scz*cos0x*sin0z+shzy*scz*sin0x*sin0y*sin0z

    x9 = scz*cos0x*cos0y-shzy*scz*sin0x*cos0z+shzx*scz*cos0x*sin0y*cos0z+shzx*scz*sin0x*sin0z+shzy*scz*cos0x*sin0y*sin0z

    x10 = tx
    x11 = ty
    x12 = tz

    x1 = tf.expand_dims(x1, 1)
    x2 = tf.expand_dims(x2, 1)
    x3 = tf.expand_dims(x3, 1)
    x4 = tf.expand_dims(x4, 1)
    x5 = tf.expand_dims(x5, 1)
    x6 = tf.expand_dims(x6, 1)
    x7 = tf.expand_dims(x7, 1)
    x8 = tf.expand_dims(x8, 1)
    x9 = tf.expand_dims(x9, 1)
    x10 = tf.expand_dims(x10, 1)
    x11 = tf.expand_dims(x11, 1)
    x12 = tf.expand_dims(x12, 1)

    array = tf.concat([ x1,x2,x3,x10,
                        x4,x5,x6,x11,
                        x7,x8,x9,x12,], 1)

    # print(" >>>>>>>>>> matrix: "+str(array.shape))   
    return array

def affine_flow(tensors):
    # print(" >>>>>>>>>> affine_flow_layer")
    imgs = tensors[0]
    array = tensors[1]
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # print(" >>>>>>>>>> array: "+str(array.shape))
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


def affine_flow_3D(tensors):

    imgs = tensors[0]
    array = tensors[1]
    print(imgs.shape)
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    zlen = tf.shape(imgs)[4]

    grids = batch_mgrid(n_batch, xlen, ylen, zlen)
    grids = tf.reshape(grids, [n_batch, 3, -1])


    theta = tf.reshape(array, [-1, 3, 4])
    matrix = tf.slice(theta, [0, 0, 0], [-1, -1, 3])
    t = tf.slice(theta, [0, 0, 3], [-1, -1, -1])

    T_g = tf.matmul(matrix, grids) + t
    T_g = tf.reshape(T_g, [n_batch, 3, xlen, ylen, zlen])
    output = batch_warp3d(imgs, T_g)
    return output

def affine_flow_output_shape(input_shapes):

    shape1 = list(input_shapes[0])
    return (shape1[0],1,shape1[2],shape1[3])

def affine_flow_3D_output_shape(input_shapes):

    shape1 = list(input_shapes[0])
    return (shape1[0], shape1[2],shape1[3],shape1[4])


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
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # imgs = tf.transpose(imgs,[0,2,3,1])
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat)
    return output
    
def batch_warp3d(imgs, mappings):

    n_batch = tf.shape(imgs)[0]
    coords = tf.reshape(mappings, [n_batch, 3, -1])
    x_coords = tf.slice(coords, [0, 0, 0], [-1, 1, -1])
    y_coords = tf.slice(coords, [0, 1, 0], [-1, 1, -1])
    z_coords = tf.slice(coords, [0, 2, 0], [-1, 1, -1])
    x_coords_flat = tf.reshape(x_coords, [-1])
    y_coords_flat = tf.reshape(y_coords, [-1])
    z_coords_flat = tf.reshape(z_coords, [-1])
    output = _interpolate3d(imgs, x_coords_flat, y_coords_flat, z_coords_flat)
    return output

def batch_displacement_warp3d(tensors):
    print(" batch_displacement_warp3d")
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
    
    output = batch_warp3d(imgs, vector_fields)
    return output

def warp_3d_layer_output_shape(input_shapes):
  
    shape1 = list(input_shapes[0])
    # print(" >>>>>>>>>> shape1: "+str(tuple(shape1).shape))
    return tuple(shape1)
    
def _interpolate2d(imgs, x, y):
    # print("interpolate2d")

    n_batch = tf.shape(imgs)[0]
    n_channel = tf.shape(imgs)[1]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
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

def _interpolate3d(imgs, x, y, z):

    n_batch = tf.shape(imgs)[0]
    n_channel = tf.shape(imgs)[1]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]
    zlen = tf.shape(imgs)[4]
    
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    z = tf.cast(z, tf.float32)
    xlen_f = tf.cast(xlen, tf.float32)
    ylen_f = tf.cast(ylen, tf.float32)
    zlen_f = tf.cast(zlen, tf.float32)
    zero = tf.zeros([], dtype='int32')
    max_x = tf.cast(xlen - 1, 'int32')
    max_y = tf.cast(ylen - 1, 'int32')
    max_z = tf.cast(zlen - 1, 'int32')

    # scale indices from [-1, 1] to [0, xlen/ylen]
    x = (x + 1.) * (xlen_f - 1.) * 0.5
    y = (y + 1.) * (ylen_f - 1.) * 0.5
    z = (z + 1.) * (zlen_f - 1.) * 0.5

    # do sampling
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), 'int32')
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    def _repeat(base_indices, n_repeats):
        base_indices = tf.matmul(
            tf.reshape(base_indices, [-1, 1]),
            tf.ones([1, n_repeats], dtype='int32'))
        return tf.reshape(base_indices, [-1])

    base = _repeat(tf.range(n_batch) * xlen * ylen * zlen,
                   xlen * ylen * zlen)
    base_x0 = base + x0 * ylen * zlen
    base_x1 = base + x1 * ylen * zlen
    base00 = base_x0 + y0 * zlen
    base01 = base_x0 + y1 * zlen
    base10 = base_x1 + y0 * zlen
    base11 = base_x1 + y1 * zlen
    index000 = base00 + z0
    index001 = base00 + z1
    index010 = base01 + z0
    index011 = base01 + z1
    index100 = base10 + z0
    index101 = base10 + z1
    index110 = base11 + z0
    index111 = base11 + z1

    # use indices to lookup pixels in the flat image and restore
    # n_channel dim
    imgs_flat = tf.reshape(imgs, [-1, n_channel])
    imgs_flat = tf.cast(imgs_flat, tf.float32)
    I000 = tf.gather(imgs_flat, index000)
    I001 = tf.gather(imgs_flat, index001)
    I010 = tf.gather(imgs_flat, index010)
    I011 = tf.gather(imgs_flat, index011)
    I100 = tf.gather(imgs_flat, index100)
    I101 = tf.gather(imgs_flat, index101)
    I110 = tf.gather(imgs_flat, index110)
    I111 = tf.gather(imgs_flat, index111)

    # and finally calculate interpolated values
    dx = x - tf.cast(x0, tf.float32)
    dy = y - tf.cast(y0, tf.float32)
    dz = z - tf.cast(z0, tf.float32)
    w000 = tf.expand_dims((1. - dx) * (1. - dy) * (1. - dz), 1)
    w001 = tf.expand_dims((1. - dx) * (1. - dy) * dz, 1)
    w010 = tf.expand_dims((1. - dx) * dy * (1. - dz), 1)
    w011 = tf.expand_dims((1. - dx) * dy * dz, 1)
    w100 = tf.expand_dims(dx * (1. - dy) * (1. - dz), 1)
    w101 = tf.expand_dims(dx * (1. - dy) * dz, 1)
    w110 = tf.expand_dims(dx * dy * (1. - dz), 1)
    w111 = tf.expand_dims(dx * dy * dz, 1)
    output = tf.add_n([w000 * I000, w001 * I001, w010 * I010, w011 * I011,
                       w100 * I100, w101 * I101, w110 * I110, w111 * I111])

    # reshape
    print(output.shape)
    output = tf.reshape(output, [n_batch, n_channel, xlen, ylen, zlen])
    print(output.shape)
    # output = tf.reshape(output, [n_batch, xlen, ylen, zlen])
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

        # print(" >>>>>>>>>> Mapping_7_Affine_Para_layer")
        imgs = tensors[0]
        Squeezed_Affine_Para = tensors[1]
        # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
        # print(" >>>>>>>>>> Squeezed_Affine_Para: "+str(Squeezed_Affine_Para.shape))
        n_batch = tf.shape(imgs)[0]
        

        sx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,0], [n_batch, 1]), 1) 
        sy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,1], [n_batch, 1]), 1) 
        cx = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,2], [n_batch, 1]), 1) 
        cy = tf.squeeze(tf.slice(Squeezed_Affine_Para, [0,3], [n_batch, 1]), 1) 
        theta = tf.slice(Squeezed_Affine_Para, [0,4], [n_batch, 1]) 
        theta = tf.squeeze(theta, 1)
        # print(" >>>>>>>>>> theta: "+str(theta.shape))
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

        # print(" >>>>>>>>>> 1.sx: "+str(sx.shape))
        # print(" >>>>>>>>>> 2.sy: "+str(sy.shape))
        # print(" >>>>>>>>>> 3.cx: "+str(sx.shape))
        # print(" >>>>>>>>>> 4.cy: "+str(sy.shape))
        # print(" >>>>>>>>>> 5.theta: "+str(theta.shape))
        # print(" >>>>>>>>>> 5. cos0: "+str(cos0.shape))
        # print(" >>>>>>>>>> 5. sin0: "+str(sin0.shape))
        # print(" >>>>>>>>>> 6.tx: "+str(tx.shape))
        # print(" >>>>>>>>>> 7.ty: "+str(ty.shape))

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
        # print(" >>>>>>>>>> Affine_Para_7: "+str(Affine_Para_7.shape))

        return Affine_Para_7


def non_linear_warp_2d(tensors):
    # print(" non_linear_warp_2d")
    imgs = tensors[0]
    vector_fields = tensors[1]
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # print(" >>>>>>>>>> vector_fields: "+str(vector_fields.shape))
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[2]
    ylen = tf.shape(imgs)[3]

    grids = non_linear_mgrid(n_batch, xlen, ylen)
    # print(" >>>>>>>>>> grids: "+str(grids.shape))
    
    T_g = grids + vector_fields
    # print(" >>>>>>>>>> T_g: "+str(T_g.shape))
    
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
    # print("non_linear_warp2d")
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))
    # # imgs = tf.transpose(imgs,[0,2,3,1])
    # print(" >>>>>>>>>> imgs: "+str(imgs.shape))

    output = _interpolate2d(imgs, x_coords_flat, y_coords_flat)
    return output

def _repeat(base_indices, n_repeats):
    base_indices = tf.matmul(
        tf.reshape(base_indices, [-1, 1]),
        tf.ones([1, n_repeats], dtype='int32'))
    return tf.reshape(base_indices, [-1])

def _interpolate2d(imgs, x, y):
    # print("interpolate2d")

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
