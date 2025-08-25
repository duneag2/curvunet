import os
from glob import glob
from tqdm import tqdm
import numpy as np
import nibabel as nib
import cv2
from pathlib import Path

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def join_paths(*args):
    return os.path.join(*args)

def read_nii(filepath):
    ct_scan = nib.load(filepath).get_fdata()
    return np.rot90(np.array(ct_scan))  # 회전해서 정렬 맞춤

def clip_scan(img, min_value, max_value):
    return np.clip(img, min_value, max_value)

def linear_scale(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img * 255

def resize_image(image, height, width, method):
    return cv2.resize(image, (width, height), interpolation=method)

def resize_scan(scan, new_height, new_width, scan_type):
    resized_scan = np.zeros((new_height, new_width, scan.shape[2]), dtype=np.uint8)
    method = cv2.INTER_CUBIC if scan_type == "image" else cv2.INTER_NEAREST
    for i in range(scan.shape[2]):
        resized_scan[:, :, i] = resize_image(scan[:, :, i], new_height, new_width, method)
    return resized_scan

def save_unet3p_images(scan, save_path, base_name):
    for i in range(scan.shape[2]):
        before = i - 1 if i > 0 else 0
        after = i + 1 if i < scan.shape[2] - 1 else scan.shape[2] - 1

        triplet = np.stack((scan[:, :, before], scan[:, :, i], scan[:, :, after]), axis=-1)
        triplet = cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR)
        out_path = join_paths(save_path, f"image_{base_name}_{i}.png")
        cv2.imwrite(out_path, triplet)

def save_unet3p_mask(scan, save_path, base_name):
    scan = np.where(scan != 0, 1, 0).astype(np.uint8) * 255
    for i in range(scan.shape[2]):
        out_path = join_paths(save_path, f"mask_{base_name}_{i}.png")
        cv2.imwrite(out_path, scan[:, :, i])

def process_scan(image_path, mask_path, image_out_dir, mask_out_dir,
                 height=320, width=320, min_val=-200, max_val=250):
    
    base_name = Path(image_path).name.replace("_0000.nii.gz", "").replace("new.nii.gz", "")
    if base_name.startswith("image_"):
        base_name = base_name[len("image_"):]

    img = read_nii(image_path)
    msk = read_nii(mask_path)

    if img.shape != msk.shape:
        print(f"[Shape mismatch] Skipping: {base_name}")
        return

    img = clip_scan(img, min_val, max_val)
    img = linear_scale(img).astype(np.uint8)
    img = resize_scan(img, height, width, "image")
    msk = resize_scan(msk, height, width, "mask")

    save_unet3p_images(img, image_out_dir, base_name)
    save_unet3p_mask(msk, mask_out_dir, base_name)

# def run_preprocessing(anam1_dir, anam2_dir, output_root):
#     img_out = join_paths(output_root, "images")
#     mask_out = join_paths(output_root, "mask")
#     create_directory(img_out)
#     create_directory(mask_out)

#     # 안암 1
#     anam1_img_paths = sorted(glob(join_paths(anam1_dir, "imagesTr", "*.nii.gz")))
#     for img_path in tqdm(anam1_img_paths, desc="안암 1 처리"):
#         mask_path = img_path.replace("imagesTr", "maskTr").replace("_0000", "")
#         if os.path.exists(mask_path):
#             process_scan(img_path, mask_path, img_out, mask_out)

#     # 안암 2
#     anam2_img_paths = sorted(glob(join_paths(anam2_dir, "images", "*.nii.gz")))
#     for img_path in tqdm(anam2_img_paths, desc="안암 2 처리"):
#         mask_path = img_path.replace("images", "mask").replace("_0000", "")
#         if os.path.exists(mask_path):
#             process_scan(img_path, mask_path, img_out, mask_out)

def run_preprocessing(heart_mri_root, output_root, output_root2):
    img_out = join_paths(output_root, "images")
    mask_out = join_paths(output_root, "mask")
    img_out2 = join_paths(output_root2, "images")
    mask_out2 = join_paths(output_root2, "mask")
    create_directory(img_out)
    create_directory(mask_out)
    create_directory(img_out2)
    create_directory(mask_out2)


    # Post-ablation
    post_img_paths = sorted(glob(join_paths(heart_mri_root, "images_imperial_postablation", "*.nii.gz")))
    for img_path in tqdm(post_img_paths, desc="Post-ablation 처리"):
        mask_name = Path(img_path).name.replace("_0000.nii.gz", "new.nii.gz")
        mask_path = join_paths(heart_mri_root, "mask_postablation", mask_name)
        if os.path.exists(mask_path):
            process_scan(img_path, mask_path, img_out, mask_out)

    # Pre-ablation
    pre_img_paths = sorted(glob(join_paths(heart_mri_root, "images_imperial_preablation", "*.nii.gz")))
    for img_path in tqdm(pre_img_paths, desc="Pre-ablation 처리"):
        mmask_name = Path(img_path).name.replace("_0000.nii.gz", "new.nii.gz")
        mask_path = join_paths(heart_mri_root, "mask_preablation", mask_name)
        if os.path.exists(mask_path):
            process_scan(img_path, mask_path, img_out2, mask_out2)


# 실행 예시
if __name__ == "__main__":
    run_preprocessing("/mnt/data/_segmentation_2025/심장_MRI/", "/mnt/data/_segmentation_2025/심장_MRI/imperial_post", "/mnt/data/_segmentation_2025/심장_MRI/imperial_pre")
