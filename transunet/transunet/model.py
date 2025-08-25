import transunet.encoder_layers as encoder_layers
import transunet.decoder_layers as decoder_layers
from transunet.resnet_v2 import  resnet_embeddings
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import transunet.utils as utils
import tensorflow as tf
import math
import sys

tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

MODELS_URL = 'https://storage.googleapis.com/vit_models/imagenet21k/'
        
def load_pretrained(model, fname='R50+ViT-B_16.npz'):
    """Load model weights for a known configuration."""
    origin = MODELS_URL + fname
    local_filepath = tf.keras.utils.get_file(fname, origin, cache_subdir="weights")
    utils.load_weights_numpy(model, local_filepath)
    
def TransUNet(image_size=224, 
                patch_size=16, 
                hybrid=True,
                grid=(14,14), 
                hidden_size=768,
                n_layers=12,
                n_heads=12,
                mlp_dim=3072,
                dropout=0.1,
                decoder_channels=[256,128,64,16],
                n_skip=3,
                num_classes=2,
                final_act='sigmoid',
                pretrain=True,
                freeze_enc_cnn=True,
                name='TransUNet'):
    # Tranformer Encoder
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    # Embedding
    if hybrid:
        grid_size = grid
        patch_size = image_size // 16 // grid_size[0]
        if patch_size == 0:
            patch_size = 1

        resnet50v2, features = resnet_embeddings(x, image_size=image_size, n_skip=n_skip, pretrain=pretrain)
        if freeze_enc_cnn:
            resnet50v2.trainable = False
        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
    else:
        y = x
        features = None

    y = tfkl.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True
    )(y)
    y = tfkl.Reshape(
        (y.shape[1] * y.shape[2], hidden_size))(y)
    y = encoder_layers.AddPositionEmbs(
        name="Transformer/posembed_input", trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            trainable=True
        )(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

    # Decoder CUP
    if len(decoder_channels):
        y = decoder_layers.DecoderCup(decoder_channels=decoder_channels, n_skip=n_skip)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

    # Build Model
    model =  tfk.models.Model(inputs=x, outputs=y, name=name)
    
    # Load Pretrain Weights
    if pretrain:
        load_pretrained(model)
        
    return model

sys.path.append('/home/dragon1/seungeun2025/TransUNet-tf/unet/utils')
from utils.curvature import *

from tensorflow.keras import layers as L
    
def curv_light_cnn(x, width_mult=1.0, out_ch=768, name="curv_light"):
    """
    x: (B, 224, 224, 3)  텐서
    return: (B, 14, 14, 768) 텐서
    """
    wm = width_mult
    ch1, ch2, ch3, ch4 = [int(c*wm) for c in (16, 32, 64, 128)]

    def ds_block(z, ch, s, bn=True, act=True, n=None):
        # Depthwise Separable Conv: 매우 가벼움
        z = L.DepthwiseConv2D(3, strides=s, padding="same", use_bias=not bn,
                              name=f"{n}_dw")(z)
        if bn: z = L.BatchNormalization(name=f"{n}_dw_bn")(z)
        if act: z = L.ReLU(name=f"{n}_dw_relu")(z)

        z = L.Conv2D(ch, 1, padding="same", use_bias=not bn, name=f"{n}_pw")(z)
        if bn: z = L.BatchNormalization(name=f"{n}_pw_bn")(z)
        if act: z = L.ReLU(name=f"{n}_pw_relu")(z)
        return z

    with tf.name_scope(name):
        # 224 -> 112
        x = ds_block(x, ch1, s=2, n=f"{name}_s2")
        # 112 -> 56
        x = ds_block(x, ch2, s=2, n=f"{name}_s4")
        # 56 -> 28
        x = ds_block(x, ch3, s=2, n=f"{name}_s8")
        # 28 -> 14
        x = ds_block(x, ch4, s=2, n=f"{name}_s16")

        # 가벼운 컨텍스트 확장(선택): DW 3×3 (stride 1)
        x = L.DepthwiseConv2D(3, padding="same", name=f"{name}_ctx_dw")(x)
        x = L.BatchNormalization(name=f"{name}_ctx_dw_bn")(x)
        x = L.ReLU(name=f"{name}_ctx_dw_relu")(x)

        # 최종 768 채널로 투영
        x = L.Conv2D(out_ch, 1, padding="same", name=f"{name}_proj")(x)
        x = L.BatchNormalization(name=f"{name}_proj_bn")(x)
        x = L.ReLU(name=f"{name}_proj_relu")(x)

    return x

def TransUNet_curvature(image_size=224, 
                patch_size=16, 
                hybrid=True,
                grid=(14,14), 
                hidden_size=768,
                n_layers=12,
                n_heads=12,
                mlp_dim=3072,
                dropout=0.1,
                decoder_channels=[256,128,64,16],
                n_skip=3,
                num_classes=2,
                final_act='sigmoid',
                pretrain=True,
                freeze_enc_cnn=True,
                name='TransUNet',
                curvature=None):
    # Tranformer Encoder
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    def which_curv_keras(x):
        if curvature == '2d':
            # 함수형 그래프를 안전하게: Lambda로 래핑 (curvature_2d_sobel이 tf 연산이면 바로 써도 됨)
            return tf.keras.layers.Lambda(lambda t: curvature_2d_sobel(t), name="curv_2d")(x)
        elif curvature == '3dg':
            def _f(t):
                _, g = compute_3d_curvatures(t)
                return g
            return tf.keras.layers.Lambda(_f, name="curv_3dg")(x)
        elif curvature == '3dm':
            def _f(t):
                m, _ = compute_3d_curvatures(t)
                return m
            return tf.keras.layers.Lambda(_f, name="curv_3dm")(x)
        
    # Embedding
    if hybrid:
        grid_size = grid
        patch_size = image_size // 16 // grid_size[0]
        if patch_size == 0:
            patch_size = 1

        resnet50v2, features = resnet_embeddings(x, image_size=image_size, n_skip=n_skip, pretrain=pretrain)
        if freeze_enc_cnn:
            resnet50v2.trainable = False
        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
        
    else:
        y = x
        features = None

    y = tfkl.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True
    )(y)

    # print(y.shape)


    y_curv = curv_light_cnn(which_curv_keras(x), width_mult=1.0, out_ch=768)  # (B,14,14,768)

    alpha_logit = tf.Variable(0.0, trainable=True, name="mix_alpha")
    alpha = tf.sigmoid(alpha_logit)
    y = (1 - alpha) * y + alpha * y_curv
    
    y = tfkl.Reshape(
        (y.shape[1] * y.shape[2], hidden_size))(y)
    y = encoder_layers.AddPositionEmbs(
        name="Transformer/posembed_input", trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            trainable=True
        )(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

    # Decoder CUP
    if len(decoder_channels):
        y = decoder_layers.DecoderCup(decoder_channels=decoder_channels, n_skip=n_skip)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

    # Build Model
    model =  tfk.models.Model(inputs=x, outputs=y, name=name)
    
    # Load Pretrain Weights
    if pretrain:
        load_pretrained(model)
        
    return model


def TransUNet_fused(image_size=224, 
                patch_size=16, 
                hybrid=True,
                grid=(14,14), 
                hidden_size=768,
                n_layers=12,
                n_heads=12,
                mlp_dim=3072,
                dropout=0.1,
                decoder_channels=[256,128,64,16],
                n_skip=3,
                num_classes=2,
                final_act='sigmoid',
                pretrain=True,
                freeze_enc_cnn=True,
                name='TransUNet'):
    # Tranformer Encoder
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))

    def every_curv_keras(x):
        def _f(t):
            _, g = compute_3d_curvatures(t)
            return g
        
        def _f(t):
            m, _ = compute_3d_curvatures(t)
            return m
        
        curv_2d = tf.keras.layers.Lambda(lambda t: curvature_2d_sobel(t), name="curv_2d")(x)
        curv_3dg = tf.keras.layers.Lambda(_f, name="curv_3dg")(x)
        curv_3dm = tf.keras.layers.Lambda(_f, name="curv_3dm")(x)

        return curv_2d, curv_3dg, curv_3dm
        
    # Embedding
    if hybrid:
        grid_size = grid
        patch_size = image_size // 16 // grid_size[0]
        if patch_size == 0:
            patch_size = 1

        resnet50v2, features = resnet_embeddings(x, image_size=image_size, n_skip=n_skip, pretrain=pretrain)
        if freeze_enc_cnn:
            resnet50v2.trainable = False
        y = resnet50v2.get_layer("conv4_block6_preact_relu").output
        x = resnet50v2.input
        
    else:
        y = x
        features = None

    y = tfkl.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
        trainable=True
    )(y)

    # print(y.shape)

    curv_2d, curv_3dg, curv_3dm = every_curv_keras(x)
    # 곡률 맵 합치기
    concat = tfkl.Concatenate(name="curv_concat")([curv_2d, curv_3dg, curv_3dm])
    
    # 위치별 softmax 가중치 계산 (출력 채널 수 = 곡률맵 개수)
    gating_weights = tfkl.Conv2D(
        3, (1,1), activation="softmax", name="curv_gate_conv"
    )(concat)
    
    # element-wise 곱 후 합산 (가중합)
    curv_fused = (
          curv_2d  * gating_weights[..., 0:1]
        + curv_3dg * gating_weights[..., 1:2]
        + curv_3dm * gating_weights[..., 2:3]
    )

    y_curv = curv_light_cnn(curv_fused, width_mult=1.0, out_ch=768)  # (B,14,14,768)

    alpha_logit = tf.Variable(0.0, trainable=True, name="mix_alpha")
    alpha = tf.sigmoid(alpha_logit)
    y = (1 - alpha) * y + alpha * y_curv
    
    y = tfkl.Reshape(
        (y.shape[1] * y.shape[2], hidden_size))(y)
    y = encoder_layers.AddPositionEmbs(
        name="Transformer/posembed_input", trainable=True)(y)

    y = tfkl.Dropout(0.1)(y)

    # Transformer/Encoder
    for n in range(n_layers):
        y, _ = encoder_layers.TransformerBlock(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            trainable=True
        )(y)
    y = tfkl.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)

    n_patch_sqrt = int(math.sqrt(y.shape[1]))

    y = tfkl.Reshape(
        target_shape=[n_patch_sqrt, n_patch_sqrt, hidden_size])(y)

    # Decoder CUP
    if len(decoder_channels):
        y = decoder_layers.DecoderCup(decoder_channels=decoder_channels, n_skip=n_skip)(y, features)

    # Segmentation Head
    y = decoder_layers.SegmentationHead(num_classes=num_classes, final_act=final_act)(y)

    # Build Model
    model =  tfk.models.Model(inputs=x, outputs=y, name=name)
    
    # Load Pretrain Weights
    if pretrain:
        load_pretrained(model)
        
    return model
