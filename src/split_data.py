import os
import shutil
import random
from . import config

def split_data_physically():
    print("\nTAHAP 2: PEMISAHAN DATA STRATIFIKASI (SPLITTING)")
    
    if os.path.exists(os.path.join(config.PROCESSED_DATA_DIR, 'train')):
        print("-> Data sudah pernah dipisah. Melewati tahap ini agar efisien.")
        return

    classes = sorted([d for d in os.listdir(config.RAW_DATA_DIR) 
               if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))])

    for cls in classes:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(config.PROCESSED_DATA_DIR, split, cls), exist_ok=True)

        src_cls_path = os.path.join(config.RAW_DATA_DIR, cls)
        images = os.listdir(src_cls_path)
        random.seed(config.SEED)
        random.shuffle(images)

        # Pemotongan per kelas (Stratified)
        train_idx = int(len(images) * config.TRAIN_RATIO)
        val_idx = train_idx + int(len(images) * config.VAL_RATIO)

        train_imgs = images[:train_idx]
        val_imgs = images[train_idx:val_idx]
        test_imgs = images[val_idx:]

        def copy_files(img_list, split_name):
            for img in img_list:
                src_path = os.path.join(src_cls_path, img)
                dest_path = os.path.join(config.PROCESSED_DATA_DIR, split_name, cls, img)
                shutil.copy(src_path, dest_path)

        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')
        
        print(f"-> Folder '{cls}' tersalin: {len(train_imgs)} Train | {len(val_imgs)} Val | {len(test_imgs)} Test")

    print("Semua gambar sukses dipisah ke folder data/processed!")