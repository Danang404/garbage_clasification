import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

def run_eda(dataset_dir):
    print("\nTAHAP 1: EXPLORATORY DATA ANALYSIS (EDA) & CLASS BALANCING")
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Folder data mentah tidak ditemukan di {dataset_dir}")
        return None

    # Ambil list kelas secara berurutan
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_counts = {c: len(os.listdir(os.path.join(dataset_dir, c))) for c in classes}

    # --- VISUALISASI 1: Bar Chart Distribusi ---
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), hue=list(class_counts.keys()), palette='viridis', legend=False)
    plt.title('Distribusi Data per Kelas')
    plt.xlabel('Kategori Sampah')
    plt.ylabel('Jumlah Gambar')
    plt.xticks(rotation=45, ha='right')

    # --- VISUALISASI 2: Pie Chart Proporsi ---
 # --- VISUALISASI 2: Pie Chart Proporsi ---
    plt.subplot(1, 2, 2)
    # Kita ambil daftar warna dari seaborn agar senada dengan bar chart
    pie_colors = sns.color_palette('viridis', len(class_counts))
    plt.pie(list(class_counts.values()), labels=list(class_counts.keys()), autopct='%1.1f%%', startangle=140, colors=pie_colors)
    plt.title('Proporsi Ketimpangan Data')
    plt.tight_layout()
    plt.show()

    # --- VISUALISASI 3: Sampel Gambar ---
    print("\nMenampilkan sampel gambar...")
    plt.figure(figsize=(12, 8))
    for i, cls in enumerate(classes[:12]): # Menampilkan maks 12 kelas
        img_dir = os.path.join(dataset_dir, cls)
        img_name = random.choice(os.listdir(img_dir))
        img_path = os.path.join(img_dir, img_name)
        
        img = mpimg.imread(img_path)
        plt.subplot(3, 4, i + 1)
        plt.imshow(img)
        plt.title(f"{cls}\n{img.shape}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # --- MENGHITUNG CLASS WEIGHTS UNTUK IMBALANCE ---
    print("\nMenghitung Class Weights untuk mengatasi Imbalance Data...")
    labels = []
    for i, c in enumerate(classes):
        labels.extend([i] * class_counts[c])
        
    cw_array = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = {i: weight for i, weight in enumerate(cw_array)}
    
    print(f"Total kelas: {len(classes)}")
    print("Bobot per kelas (Class Weights) berhasil dihitung!")
    
    return class_weights