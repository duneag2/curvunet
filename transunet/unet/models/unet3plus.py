"""
UNet3+ base model
"""
import tensorflow as tf
import tensorflow.keras as k
from .unet3plus_utils import conv_block


def unet3plus_default(input_shape, output_channels):
    """ UNet3+ base model """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(
        shape=input_shape,
        name="input_layer"
    )  # 320*320*3
    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512

    # block 5
    # bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

    e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

    # last layer does not have batchnorm and relu
    d = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)

    output = k.activations.softmax(d)
    
    # print('output: ', output.shape)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus')



from utils.curvature import *

def unet3plus_curvature(input_shape, output_channels, curvature):
    """ UNet3+ base model """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=input_shape, name="input_layer")  # 320*320*3
    if curvature == '2d':
        input_curvature = tf.stop_gradient(curvature_2d_sobel(input_layer))
    elif curvature == '3dg':
        _, input_curvature = tf.stop_gradient(compute_3d_curvatures(input_layer))
    elif curvature == '3dm':
        input_curvature, _ = tf.stop_gradient(compute_3d_curvatures(input_layer))
#     print('input', input_curvature.shape)
    """ Encoder """
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64
    c1 = conv_block(input_curvature, filters[0])

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128
    
    c2 = k.layers.MaxPool2D(pool_size=(2, 2))(c1)  # 160*160*64
    c2 = conv_block(c2, filters[1])  # 160*160*128

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256
    
    c3 = k.layers.MaxPool2D(pool_size=(2, 2))(c2)  # 80*80*128
    c3 = conv_block(c3, filters[2])  # 80*80*256

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512
    
    c4 = k.layers.MaxPool2D(pool_size=(2, 2))(c3)  # 40*40*256
    c4 = conv_block(c4, filters[3])  # 40*40*512

    # block 5
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024
    
    c5 = k.layers.MaxPool2D(pool_size=(2, 2))(c4)  # 20*20*512
    c5 = conv_block(c5, filters[4])  # 20*20*1024
   

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

    e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64
    
    c5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c5)  # 80*80*256  --> 40*40*256
    c5_d4 = conv_block(c5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4, c4, c5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3, c3])
    d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2, c2])
    d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, c1])
    d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

    # last layer does not have batchnorm and relu
    d = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)

    output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus_Curvature')

# fusion of 3 curvature maps

def conv_block(x, filters, n=2, is_bn=True, is_relu=True):
    for _ in range(n):
        x = k.layers.Conv2D(filters, (3, 3), padding='same')(x)
        if is_bn:
            x = k.layers.BatchNormalization()(x)
        if is_relu:
            x = k.layers.ReLU()(x)
    return x

def gated_fusion(curvatures, name="gating"):
    concat = k.layers.Concatenate(name=name+"_concat")(curvatures)
    gating_weights = k.layers.Conv2D(len(curvatures), (1, 1), activation='softmax', name=name+"_conv")(concat)
    gated = [k.layers.Multiply(name=f"{name}_mul_{i}")([curvatures[i], gating_weights[..., i:i+1]]) for i in range(len(curvatures))]
    return gated

def repeat_channels(x, target_channels):
    input_channels = x.shape[-1]
    
    repeat_times = target_channels // input_channels
    remaining_channels = target_channels % input_channels

    # Repeat the channels for the base times
    repeated_x = k.layers.Concatenate(axis=-1)([x] * repeat_times)

    # If there's a remainder, repeat the initial channels again to match the exact target
    if remaining_channels > 0:
        remaining_x = x[:, :, :, :remaining_channels]
        repeated_x = k.layers.Concatenate(axis=-1)([repeated_x, remaining_x])
    
    return repeated_x


def unet3plus_fused(input_shape, output_channels):
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=input_shape, name="input_layer")  # 320*320*3

#     # Curvature maps
#     curv_2d = tf.stop_gradient(curvature_2d_sobel(input_layer))
#     curv_3dg, curv_3dm = tf.stop_gradient(compute_3d_curvatures(input_layer))

#     # Gating mechanism
#     mu_1, mu_2, mu_3 = gated_fusion([curv_2d, curv_3dg, curv_3dm], name="curv_gating")

#     # Add mu to each curvature
#     curv_2d_enh = k.layers.Add(name="curv_2d_enh")([curv_2d, mu_1])
#     curv_3dg_enh = k.layers.Add(name="curv_3dg_enh")([curv_3dg, mu_2])
#     curv_3dm_enh = k.layers.Add(name="curv_3dm_enh")([curv_3dm, mu_3])

#     # Fuse curvature maps for encoder
#     curvature_all = k.layers.Concatenate(name="curvature_concat")([curv_2d_enh, curv_3dg_enh, curv_3dm_enh])

    # Curvature maps
    curv_2d = tf.stop_gradient(curvature_2d_sobel(input_layer))
    curv_3dg, curv_3dm = tf.stop_gradient(compute_3d_curvatures(input_layer))

    # Gating mechanism
    mu_1, mu_2, mu_3 = gated_fusion([curv_2d, curv_3dg, curv_3dm], name="curv_gating")

    # Add mu to each curvature using simple '+' to reduce graph complexity
    curv_2d_enh = curv_2d + mu_1
    curv_3dg_enh = curv_3dg + mu_2
    curv_3dm_enh = curv_3dm + mu_3

    # Fuse curvature maps for encoder
    curvature_all = k.layers.Concatenate(name="curvature_concat")([curv_2d_enh, curv_3dg_enh, curv_3dm_enh])


    # Encoder
    e1 = conv_block(input_layer, filters[0])
    c1 = conv_block(curvature_all, filters[0])

    e2 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(e1), filters[1])
    c2 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(c1), filters[1])

    e3 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(e2), filters[2])
    c3 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(c2), filters[2])

    e4 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(e3), filters[3])
    c4 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(c3), filters[3])

    e5 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(e4), filters[4])
    c5 = conv_block(k.layers.MaxPool2D(pool_size=(2, 2))(c4), filters[4])

    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    # d4
    e1_d4 = conv_block(k.layers.MaxPool2D((8, 8))(e1), cat_channels, n=1)
    e2_d4 = conv_block(k.layers.MaxPool2D((4, 4))(e2), cat_channels, n=1)
    e3_d4 = conv_block(k.layers.MaxPool2D((2, 2))(e3), cat_channels, n=1)
    e4_d4 = conv_block(e4, cat_channels, n=1)
    e5_d4 = conv_block(k.layers.UpSampling2D((2, 2))(e5), cat_channels, n=1)
    c5_d4 = conv_block(k.layers.UpSampling2D((2, 2))(c5), cat_channels, n=1)

    d4 = conv_block(k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4, c4, c5_d4]), upsample_channels, n=1)

    # d3
    e1_d3 = conv_block(k.layers.MaxPool2D((4, 4))(e1), cat_channels, n=1)
    e2_d3 = conv_block(k.layers.MaxPool2D((2, 2))(e2), cat_channels, n=1)
    e3_d3 = conv_block(e3, cat_channels, n=1)
    e4_d3 = conv_block(k.layers.UpSampling2D((2, 2))(d4), cat_channels, n=1)
    e5_d3 = conv_block(k.layers.UpSampling2D((4, 4))(e5), cat_channels, n=1)

    d3 = conv_block(k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3, c3]), upsample_channels, n=1)
    
    # curvature min/max injection
#     curv_min = k.layers.Minimum(name="curv_min")([curv_2d, curv_3dg, curv_3dm])
#     curv_max = k.layers.Maximum(name="curv_max")([curv_2d, curv_3dg, curv_3dm])
#     curv_min_proj = repeat_channels(curv_min, upsample_channels)
#     curv_max_proj = repeat_channels(curv_max, upsample_channels)
    
    # d2
    e1_d2 = conv_block(k.layers.MaxPool2D((2, 2))(e1), cat_channels, n=1)
    e2_d2 = conv_block(e2, cat_channels, n=1)
    d3_d2 = conv_block(k.layers.UpSampling2D((2, 2))(d3), cat_channels, n=1)
    d4_d2 = conv_block(k.layers.UpSampling2D((4, 4))(d4), cat_channels, n=1)
    e5_d2 = conv_block(k.layers.UpSampling2D((8, 8))(e5), cat_channels, n=1)

    d2 = conv_block(k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2, c2]), upsample_channels, n=1)
#     d2 = k.layers.Add(name="d2_with_maxcurv")([d2, curv_max_proj])
#     d2 = k.layers.UpSampling2D(size=(2, 2))(d2)  # (320,320,C)
#     d2 = k.layers.Add(name="d2_with_maxcurv")([d2, curv_max_proj])  # Now both are (320,320,C)

    # d1
    e1_d1 = conv_block(e1, cat_channels, n=1)
    d2_d1 = conv_block(k.layers.UpSampling2D((2, 2))(d2), cat_channels, n=1)
#     d2_d1 = conv_block(d2, cat_channels, n=1)
    d3_d1 = conv_block(k.layers.UpSampling2D((4, 4))(d3), cat_channels, n=1)
    d4_d1 = conv_block(k.layers.UpSampling2D((8, 8))(d4), cat_channels, n=1)
    e5_d1 = conv_block(k.layers.UpSampling2D((16, 16))(e5), cat_channels, n=1)
    
    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, c1])
    d1 = conv_block(d1, upsample_channels, n=1)
#     d1 = k.layers.Add(name="d1_with_mincurv")([d1, curv_min_proj])

    # final output
    d = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)
#     d = k.layers.Add(name="d_with_mincurv")([d, curv_min_proj])
    output = k.activations.softmax(d)

    return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus_Fused')


# for iteration purposes

# prev_curvature = None  # 초기화

# def update_curvature(curvature, lambda_val=0.1):
#     global prev_curvature

#     if prev_curvature is None:  # 첫 epoch이면 그냥 curvature 사용
#         prev_curvature = curvature
#     else:
#         prev_curvature = prev_curvature + lambda_val * (curvature - prev_curvature)

#     return prev_curvature

# def unet3plus_iter(input_shape, output_channels, curvature):
#     """ UNet3+ base model """
#     filters = [64, 128, 256, 512, 1024]

#     input_layer = k.layers.Input(shape=input_shape, name="input_layer")  # 320*320*3
#     if curvature == '2d':
#         input_curvature = curvature_2d_sobel(input_layer)
#     elif curvature == '3dg':
#         _, input_curvature = compute_3d_curvatures(input_layer)
#     elif curvature == '3dm':
#         input_curvature, _ = compute_3d_curvatures(input_layer)
    
#     updated_curvature = update_curvature(input_curvature, lambda_val=0.1)

#     """ Encoder """
#     # block 1
#     e1 = conv_block(input_layer, filters[0])  # 320*320*64
#     c1 = conv_block(updated_curvature, filters[0])

#     # block 2
#     e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
#     e2 = conv_block(e2, filters[1])  # 160*160*128
    
#     c2 = k.layers.MaxPool2D(pool_size=(2, 2))(c1)  # 160*160*64
#     c2 = conv_block(c2, filters[1])  # 160*160*128

#     # block 3
#     e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
#     e3 = conv_block(e3, filters[2])  # 80*80*256
    
#     c3 = k.layers.MaxPool2D(pool_size=(2, 2))(c2)  # 80*80*128
#     c3 = conv_block(c3, filters[2])  # 80*80*256

#     # block 4
#     e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
#     e4 = conv_block(e4, filters[3])  # 40*40*512
    
#     c4 = k.layers.MaxPool2D(pool_size=(2, 2))(c3)  # 40*40*256
#     c4 = conv_block(c4, filters[3])  # 40*40*512

#     # block 5
#     e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
#     e5 = conv_block(e5, filters[4])  # 20*20*1024
    
#     c5 = k.layers.MaxPool2D(pool_size=(2, 2))(c4)  # 20*20*512
#     c5 = conv_block(c5, filters[4])  # 20*20*1024
   

#     """ Decoder """
#     cat_channels = filters[0]
#     cat_blocks = len(filters)
#     upsample_channels = cat_blocks * cat_channels

#     """ d4 """
#     e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
#     e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

#     e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
#     e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

#     e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
#     e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

#     e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

#     e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
#     e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64
    
#     c5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(c5)  # 80*80*256  --> 40*40*256
#     c5_d4 = conv_block(c5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

#     d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4, c4, c5_d4])
#     d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

#     """ d3 """
#     e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
#     e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

#     e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
#     e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

#     e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

#     e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
#     e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

#     e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
#     e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

#     d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3, c3])
#     d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

#     """ d2 """
#     e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
#     e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

#     e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

#     d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
#     d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

#     d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
#     d4_d2 = conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

#     e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
#     e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

#     d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2, c2])
#     d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

#     """ d1 """
#     e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

#     d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
#     d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

#     d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
#     d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

#     d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
#     d4_d1 = conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

#     e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
#     e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

#     d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, c1])
#     d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

#     # last layer does not have batchnorm and relu
#     d = conv_block(d1, output_channels, n=1, is_bn=False, is_relu=False)

#     output = k.activations.softmax(d)

#     return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet_3Plus')


# -

def tiny_unet3plus(input_shape, output_channels, training):
    """ Sample model only for testing during development """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=input_shape, name="input_layer")  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0] // 2)  # 320*320*64
    e1 = conv_block(e1, filters[0] // 2)  # 320*320*64

    # last layer does not have batch norm and relu
    d = conv_block(e1, output_channels, n=1, is_bn=False, is_relu=False)
    output = k.activations.softmax(d, )

    if training:
        e2 = conv_block(e1, filters[0] // 2)  # 320*320*64
        d2 = conv_block(e2, output_channels, n=1, is_bn=False, is_relu=False)
        output2 = k.activations.softmax(d2)
        return tf.keras.Model(inputs=input_layer, outputs=[output, output2], name='UNet3Plus')
    else:
        return tf.keras.Model(inputs=input_layer, outputs=[output], name='UNet3Plus')


if __name__ == "__main__":
    """## Model Compilation"""
    INPUT_SHAPE = [320, 320, 1]
    OUTPUT_CHANNELS = 1

    unet_3P = unet3plus(INPUT_SHAPE, OUTPUT_CHANNELS)
    unet_3P.summary()

    # tf.keras.utils.plot_model(unet_3P, show_layer_names=True, show_shapes=True)

    # unet_3P.save("unet_3P.hdf5")
