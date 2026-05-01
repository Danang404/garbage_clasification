import os
import torch
from torchvision import datasets
from torchvision.transforms import v2
from src import config

def get_dataloaders():
    print("\nTAHAP 3: MEMUAT DATALOADER (PYTORCH)")
    
    # 1. Transformasi & Augmentasi (Standar Modern PyTorch v2)
    # Model pre-trained PyTorch mewajibkan normalisasi dengan angka spesifik ImageNet
    train_transform = v2.Compose([
        v2.Resize(config.IMG_SIZE),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=20),
        v2.RandomResizedCrop(size=config.IMG_SIZE, scale=(0.8, 1.0)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validasi & Test HANYA di-resize dan di-normalisasi 
    val_test_transform = v2.Compose([
        v2.Resize(config.IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Muat dataset dari folder
    train_dataset = datasets.ImageFolder(root=os.path.join(config.PROCESSED_DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config.PROCESSED_DATA_DIR, 'val'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(config.PROCESSED_DATA_DIR, 'test'), transform=val_test_transform)

    # 3. Buat Dataloader (Pengatur Batch & Multithreading)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"-> Dataloader siap! Mendeteksi {len(class_names)} kelas.")
    
    return train_loader, val_loader, test_loader, class_names