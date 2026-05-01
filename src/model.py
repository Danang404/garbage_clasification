import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def build_model(num_classes):
    print("\nTAHAP 4: MEMBANGUN MODEL (EFFICIENTNET-V2-S PYTORCH)")
    
    # 1. Load Pre-trained Model (V2-S / Small)
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    
    # 2. Freeze semua layer dasar (Transfer Learning Awal)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Ganti Layer Klasifikasi Terakhir (Classifier Head)
    # Mengambil jumlah fitur dari layer sebelum classifier
    num_ftrs = model.classifier[1].in_features 
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_ftrs, num_classes)
    )
    
    print("Arsitektur PyTorch V2-S berhasil dibuat!")
    return model