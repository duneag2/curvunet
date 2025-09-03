import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage

# def curvature_2d_sobel(y_pred):
# #     # Compute first-order gradients
# #     G_x = tf.image.sobel_edges(y_pred)[..., 0]
# #     G_y = tf.image.sobel_edges(y_pred)[..., 1]

# #     # Compute second-order gradients
# #     G_xx = tf.image.sobel_edges(G_x)[..., 0]
# #     G_yy = tf.image.sobel_edges(G_y)[..., 1]
# #     G_xy = tf.image.sobel_edges(G_x)[..., 1]

#     # Extract G_x and G_y properly
#     G_x = G[..., 0, 0]  # x 방향 미분
#     G_y = G[..., 0, 1]  # y 방향 미분

#     # Compute second-order gradients
#     G_xx = tf.image.sobel_edges(G_x)[..., 0, 0]  # ∂²I/∂x²
#     G_yy = tf.image.sobel_edges(G_y)[..., 0, 1]  # ∂²I/∂y²
#     G_xy = tf.image.sobel_edges(G_x)[..., 0, 1]  # ∂²I/∂x∂y

#     # Calculate signed curvature (kappa)
#     numerator = (tf.square(G_x) * G_yy) + (tf.square(G_y) * G_xx) - (2 * G_x * G_y * G_xy)
#     denominator = tf.pow((tf.square(G_x) + tf.square(G_y)), 1.5) + 1e-6  # to avoid division by zero

#     kappa = numerator / denominator
#     return kappa

# # +
# # def create_3d_gaussian(label_image, sigma=1.0, slice_num=5):
# #     label_mask = tf.cast(label_image > 0, dtype=tf.float32)
    
# #     x = np.linspace(-3, 3, 320)
# #     y = np.linspace(-3, 3, 320)
# #     X, Y = np.meshgrid(x, y)
    
# #     Z = tf.convert_to_tensor(np.exp(-0.5 * (X**2 + Y**2) / sigma**2), dtype=tf.float32)
    
# #     label_mask_resized = label_mask[0, :, :, 0]
    
# #     Z = Z * tf.squeeze(label_mask_resized) 

# #     Z_3d = tf.stack([Z] * slice_num, axis=-1)

# #     return X, Y, Z_3d

# def create_3d_gaussian(label_image, sigma=1.0, slice_num=5):
    
#     label_image = label_image[np.random.randint(0, label_image.shape[0]), :, :, 0].cpu().numpy()
#     width, height = label_image.shape
#     x = np.linspace(-3, 3, width)
#     y = np.linspace(-3, 3, height)
#     X, Y = np.meshgrid(x, y)

#     Z = np.exp(-0.5 * (X**2 + Y**2) / sigma**2)
#     Z = Z * (label_image > 0)
    
#     Z_3d = np.stack([Z] * slice_num, axis=-1)
    
#     return X, Y, Z_3d


# # -

# # 3D Sobel filter
# def sobel_3d(image):
#     sobel_x = np.array([[[1, 0, -1],
#                           [2, 0, -2],
#                           [1, 0, -1]],
                         
#                          [[1, 0, -1],
#                           [2, 0, -2],
#                           [1, 0, -1]],
                         
#                          [[1, 0, -1],
#                           [2, 0, -2],
#                           [1, 0, -1]]])

#     sobel_y = np.array([[[1, 2, 1],
#                           [0, 0, 0],
#                           [-1, -2, -1]],
                         
#                          [[1, 2, 1],
#                           [0, 0, 0],
#                           [-1, -2, -1]],
                         
#                          [[1, 2, 1],
#                           [0, 0, 0],
#                           [-1, -2, -1]]])

#     sobel_z = np.array([[[1, 2, 1],
#                           [1, 2, 1],
#                           [1, 2, 1]],
                         
#                          [[0, 0, 0],
#                           [0, 0, 0],
#                           [0, 0, 0]],
                         
#                          [[-1, -2, -1],
#                           [-1, -2, -1],
#                           [-1, -2, -1]]])

#     # Apply the 3D Sobel filter
#     G_x = convolve_3d(image, sobel_x)
#     G_y = convolve_3d(image, sobel_y)
#     G_z = convolve_3d(image, sobel_z)

#     return G_x, G_y, G_z

# # 3D convolution function
# def convolve_3d(image, kernel):
#     from scipy.ndimage import convolve
#     return convolve(image, kernel, mode='constant', cval=0.0)

# def mean_curvature_3d(Z):
#     Z_tensor = tf.convert_to_tensor(Z, dtype=tf.float32)
    
#     G_x, G_y, G_z = sobel_3d(Z)

#     # Compute second-order gradients
#     G_xx, G_yy, G_zz = sobel_3d(G_x), sobel_3d(G_y), sobel_3d(G_z)
#     G_xy = sobel_3d(G_x)[1]
#     G_xz = sobel_3d(G_x)[2]
#     G_yz = sobel_3d(G_y)[2]

#     # Calculate mean curvature (H)
#     numerator = G_xx * G_yz + G_yy * G_xz + G_zz * G_xy - (G_x * G_x * G_yy + G_y * G_y * G_xx + G_z * G_z * G_zz)
#     denominator = tf.pow((tf.square(G_x) + tf.square(G_y) + tf.square(G_z)), 1.5) + 1e-6  # to avoid division by zero

#     H = numerator / denominator
#     return H


# 2D curvature function using Sobel
def curvature_2d_sobel(y_pred):
    # Compute first-order gradients
    G_x = tf.image.sobel_edges(y_pred)[..., 0]
    G_y = tf.image.sobel_edges(y_pred)[..., 1]

    # Compute second-order gradients
    G_xx = tf.image.sobel_edges(G_x)[..., 0]
    G_yy = tf.image.sobel_edges(G_y)[..., 1]
    G_xy = tf.image.sobel_edges(G_x)[..., 1]

    # Calculate signed curvature (kappa)
    numerator = (tf.square(G_x) * G_yy) + (tf.square(G_y) * G_xx) - (2 * G_x * G_y * G_xy)
    denominator = tf.pow((tf.square(G_x) + tf.square(G_y)), 1.5) + 1e-6  # to avoid division by zero

    kappa = numerator / denominator
    return kappa

def compute_3d_curvatures(volume):
    def numpy_func(volume_np):
        return compute_3d_curvatures_value(volume_np)

    mean_curvature, gaussian_curvature = tf.py_function(numpy_func, [volume], [tf.float32, tf.float32])

    batch_size = volume.shape[0]  # None이면 일단 batch_size를 None으로 유지
    height, width, ch = volume.shape[1], volume.shape[2], volume.shape[3]
    
#     print(mean_curvature)
#     print(mean_curvature.shape)
#     print(gaussian_curvature.shape)

    mean_curvature.set_shape((batch_size, height, width, ch))  # (batch, height, width, 1)
    gaussian_curvature.set_shape((batch_size, height, width, ch))
    
#     print(mean_curvature.shape)
#     print(gaussian_curvature.shape)
    
#     mean_curvature = tf.convert_to_tensor(mean_curvature, dtype=tf.float32)
#     gaussian_curvature = tf.convert_to_tensor(gaussian_curvature, dtype=tf.float32)
    
    return mean_curvature, gaussian_curvature

def compute_3d_curvatures_value(volume):
    """
    Compute Mean and Gaussian curvature for a 3D volume using Sobel gradients.
    :param volume: 3D numpy array (depth, height, width)
    :return: mean_curvature, gaussian_curvature
    """
    # Compute first-order derivatives
    Gx = scipy.ndimage.sobel(volume, axis=2, mode='constant')  # ∂f/∂x
    Gy = scipy.ndimage.sobel(volume, axis=1, mode='constant')  # ∂f/∂y
    Gz = scipy.ndimage.sobel(volume, axis=0, mode='constant')  # ∂f/∂z
    
    # Compute second-order derivatives
    Gxx = scipy.ndimage.sobel(Gx, axis=2, mode='constant')  # ∂²f/∂x²
    Gyy = scipy.ndimage.sobel(Gy, axis=1, mode='constant')  # ∂²f/∂y²
    Gzz = scipy.ndimage.sobel(Gz, axis=0, mode='constant')  # ∂²f/∂z²
    Gxy = scipy.ndimage.sobel(Gx, axis=1, mode='constant')  # ∂²f/∂x∂y
    Gxz = scipy.ndimage.sobel(Gx, axis=0, mode='constant')  # ∂²f/∂x∂z
    Gyz = scipy.ndimage.sobel(Gy, axis=0, mode='constant')  # ∂²f/∂y∂z
    
    # Compute gradient magnitude squared
    G_mag_sq = Gx**2 + Gy**2 + Gz**2
    G_mag_sq[G_mag_sq == 0] = 1e-6  # Avoid division by zero
    coeff_lambda = 1e-6
    
    # Mean curvature formula (divergence of normal vector field)
    mean_curvature = (Gxx * (Gy**2 + Gz**2) + Gyy * (Gx**2 + Gz**2) + Gzz * (Gx**2 + Gy**2)
                      - 2 * (Gx * Gy * Gxy + Gx * Gz * Gxz + Gy * Gz * Gyz)) / (2 * np.power(G_mag_sq, 1.5))
    
    # Gaussian curvature formula (determinant of Hessian)
    gaussian_curvature = ((Gxx * Gyy * Gzz) + 2 * (Gxy * Gxz * Gyz) - (Gxx * Gyz**2) - (Gyy * Gxz**2) - (Gzz * Gxy**2)) / (G_mag_sq**2 + coeff_lambda)
    
# 망가우시안
#     trace_H = Gxx + Gyy + Gzz

#     # Mean Curvature formula
#     mean_curvature = ((Gx**2 * Gxx + 2 * Gx * Gy * Gxy + 2 * Gx * Gz * Gxz + Gy**2 * Gyy + 2 * Gy * Gz * Gyz + Gz**2 * Gzz - G_mag_sq * trace_H)/ (2 * np.power(G_mag_sq, 1.5)))
    
#     gaussian_curvature = ((Gx**2 * (Gyy * Gzz - Gyz**2) + Gy**2 * (Gxx * Gzz - Gxz**2) + Gz**2 * (Gxx * Gyy - Gxy**2) + 2 * Gx * Gy * (Gyz * Gxz - Gxy * Gzz) + 2 * Gx * Gz * (Gxy * Gyz - Gxz * Gyy) + 2 * Gy * Gz * (Gxy * Gxz - Gxx * Gyz)) + coeff_lambda / G_mag_sq**2)
#     mean_curvature = np.mean(mean_curvature, axis=0) # 평균

    
#     mean_curvature = np.mean(mean_curvature, axis=0) # 평균
#     mean_curvature = np.mean(mean_curvature, axis=2)
    
#     gaussian_curvature = np.mean(gaussian_curvature, axis=0) # 평균
#     gaussian_curvature = np.mean(gaussian_curvature, axis=2)
    
#     print('mean: ', mean_curvature.shape)
#     print('gauss: ', gaussian_curvature.shape)
    
    return mean_curvature, gaussian_curvature

