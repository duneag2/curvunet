"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from utils.curvature import *


def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value, axis=0)


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_loss(y_true, y_pred):
    """
    Dice loss function.
    """
    return 1 - dice_coef(y_true, y_pred)


def curvature_2d_loss(y_true, y_pred):
    """
    Calculate 2D curvature loss between predicted and target curvatures.
    """
    true_curvature = curvature_2d_sobel(y_true)
    pred_curvature = curvature_2d_sobel(y_pred)
#     print(true_curvature.shape)
#     print(pred_curvature.shape)
#     print(tf.cast(tf.reduce_mean(tf.abs(true_curvature - pred_curvature)), tf.float32).shape)
    return tf.cast(tf.reduce_mean(tf.abs(true_curvature - pred_curvature)), tf.float32)


def curvature_3d_gaussian_loss(y_true, y_pred):
    """
    Calculate 3D Gaussian curvature loss between predicted and target curvatures.
    """
    
#     print(y_true.shape)
#     print(y_pred.shape)

    _, gaussian_true = compute_3d_curvatures_value(y_true)
    _, gaussian_pred = compute_3d_curvatures_value(y_pred)
    gaussian_true = tf.convert_to_tensor(gaussian_true)
    gaussian_pred = tf.convert_to_tensor(gaussian_pred)
    
#     print(gaussian_true.shape)
#     print(gaussian_pred.shape)
#     print(tf.cast(tf.reduce_mean(tf.abs(gaussian_true - gaussian_pred)), tf.float32))
    
    return tf.cast(tf.reduce_mean(tf.abs(gaussian_true - gaussian_pred)), tf.float32)


def curvature_3d_mean_loss(y_true, y_pred):
    """
    Calculate 3D Mean curvature loss between predicted and target curvatures.
    """
    
    mean_true, _ = compute_3d_curvatures_value(y_true)
    mean_pred, _ = compute_3d_curvatures_value(y_pred)
    mean_true = tf.convert_to_tensor(mean_true)
    mean_pred = tf.convert_to_tensor(mean_pred)
    
    return tf.cast(tf.reduce_mean(tf.abs(mean_true - mean_pred)), tf.float32)
