import os
import shutil
import random
from PIL import Image
import numpy as np

# 원본 경로
source_images_dir = '/mnt/data/kits/kits23/preprocessed_final/train/images'
source_masks_dir = '/mnt/data/kits/kits23/preprocessed_final/train/mask'

# 출력 경로
output_30p_images = '/mnt/data/kits/kits23/preprocessed_final/train_30p/images'
output_30p_masks = '/mnt/data/kits/kits23/preprocessed_final/train_30p/mask'
output_100n_images = '/mnt/data/kits/kits23/preprocessed_final/train_100n/images'
output_100n_masks = '/mnt/data/kits/kits23/preprocessed_final/train_100n/mask'

# 입력 폴더 존재 확인
if not os.path.exists(source_images_dir) or not os.path.exists(source_masks_dir):
    raise FileNotFoundError("train/images 또는 train/mask 폴더가 존재하지 않습니다.")

# 출력 폴더 생성
for path in [output_30p_images, output_30p_masks, output_100n_images, output_100n_masks]:
    os.makedirs(path, exist_ok=True)

# 유효한 이미지-마스크 쌍 추출 (마스크 기준으로 필터링)
valid_pairs = []
for filename in os.listdir(source_masks_dir):
    mask_path = os.path.join(source_masks_dir, filename)
    image_filename = filename.replace('mask', 'image')
    image_path = os.path.join(source_images_dir, image_filename)

    if not os.path.exists(image_path):
        print(f"이미지 누락: {image_filename}")
        continue

    try:
        with Image.open(mask_path) as mask_img:
            mask_array = np.array(mask_img)
            if np.all(mask_array == mask_array.flat[0]):
                continue  # 모든 픽셀이 같은 마스크는 제외
            valid_pairs.append((image_filename, filename))
    except Exception as e:
        print(f"파일 열기 실패: {filename} - {e}")

# 셔플
random.shuffle(valid_pairs)

# 30% 저장
n_30p = max(1, int(len(valid_pairs) * 0.3))
for image_file, mask_file in valid_pairs[:n_30p]:
    shutil.copy(os.path.join(source_images_dir, image_file), os.path.join(output_30p_images, image_file))
    shutil.copy(os.path.join(source_masks_dir, mask_file), os.path.join(output_30p_masks, mask_file))

# 100쌍 저장
n_100 = min(100, len(valid_pairs))
for image_file, mask_file in valid_pairs[:n_100]:
    shutil.copy(os.path.join(source_images_dir, image_file), os.path.join(output_100n_images, image_file))
    shutil.copy(os.path.join(source_masks_dir, mask_file), os.path.join(output_100n_masks, mask_file))

# 결과 출력
print(f"총 유효한 이미지-마스크 쌍: {len(valid_pairs)}")
print(f"train_30p에 저장된 쌍: {n_30p}")
print(f"train_100n에 저장된 쌍: {n_100}")
