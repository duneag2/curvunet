"""
UNet 3+ Loss
"""
from .loss import dice_loss, focal_loss, ssim_loss, iou_loss, curvature_2d_loss, curvature_3d_gaussian_loss, curvature_3d_mean_loss

def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
#     f_loss = focal_loss(y_true, y_pred)
#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     jacard_loss = iou_loss(y_true, y_pred)
    
    d_loss = dice_loss(y_true, y_pred)

    return d_loss 

def unet3p_hybrid_loss_2d(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
#     f_loss = focal_loss(y_true, y_pred)
#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     jacard_loss = iou_loss(y_true, y_pred)
    
    d_loss = dice_loss(y_true, y_pred)
    curvature_2d_sobel_loss = curvature_2d_loss(y_true, y_pred)
#     curvature_gaussian_loss = curvature_3d_gaussian_loss(y_true, y_pred)
#     curvature_mean_loss = curvature_3d_mean_loss(y_true, y_pred)

    return d_loss + curvature_2d_sobel_loss # + curvature_gaussian_loss + curvature_mean_loss

def unet3p_hybrid_loss_3dg(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
#     f_loss = focal_loss(y_true, y_pred)
#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     jacard_loss = iou_loss(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred)
#     curvature_2d_sobel_loss = curvature_2d_loss(y_true, y_pred)
    curvature_gaussian_loss = curvature_3d_gaussian_loss(y_true, y_pred)

#     curvature_mean_loss = curvature_3d_mean_loss(y_true, y_pred)

#     print(d_loss)
#     print(curvature_gaussian_loss)
#     print('sum: ', d_loss + curvature_gaussian_loss)
    scaling_factor = 1e-5
#     print(d_loss + scaling_factor * curvature_gaussian_loss)

    return d_loss + scaling_factor * curvature_gaussian_loss

def unet3p_hybrid_loss_3dm(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    Hybrid loss for segmentation in three-level hierarchy – pixel,
    patch and map-level, which is able to capture both large-scale
    and fine structures with clear boundaries.
    """
#     f_loss = focal_loss(y_true, y_pred)
#     ms_ssim_loss = ssim_loss(y_true, y_pred)
#     jacard_loss = iou_loss(y_true, y_pred)
    
    d_loss = dice_loss(y_true, y_pred)
#     curvature_2d_sobel_loss = curvature_2d_loss(y_true, y_pred)
#     curvature_gaussian_loss = curvature_3d_gaussian_loss(y_true, y_pred)
    curvature_mean_loss = curvature_3d_mean_loss(y_true, y_pred)
    scaling_factor = 1e-5

    return d_loss + scaling_factor * curvature_mean_loss
