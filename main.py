import random
import numpy as np
import torch
from src import config
from src.split_data import split_data_physically
from src.eda import run_eda
from src.data_loader import get_dataloaders
from src.model import build_model
from src.train import train_model
import mlflow

def main():
    print("="*60)
    print(" PIPELINE KLASIFIKASI SAMPAH DAUR ULANG (PYTORCH) ")
    print("="*60)

    mlflow.set_experiment("garbage-classification")

    with mlflow.start_run(run_name="full_pipeline"):
        split_data_physically()
        class_weights = run_eda(config.RAW_DATA_DIR)

        train_loader, val_loader, test_loader, class_names = get_dataloaders()
        model = build_model(num_classes=len(class_names))

        # Log config global
        mlflow.log_params({
            "model": "EfficientNet-V2-S",
            "img_size": config.IMG_SIZE,
            "batch_size": config.BATCH_SIZE,
            "num_classes": len(class_names),
            "seed": config.SEED,
        })

        print("\n" + "="*30)
        print(" FASE 1: INITIAL TRANSFER LEARNING ")
        print("="*30)
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights_dict=class_weights,
            num_classes=len(class_names),
            epochs=config.EPOCHS_INITIAL,
            learning_rate=config.LR_INITIAL,
            save_name='best_model_initial'
        )

        print("\n" + "="*30)
        print(" FASE 2: FINE TUNING ")
        print("="*30)
        for param in model.parameters():
            param.requires_grad = True

        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights_dict=class_weights,
            num_classes=len(class_names),
            epochs=config.EPOCHS_FINETUNE,
            learning_rate=config.LR_FINETUNE,
            save_name='best_model_finetuned',
            is_fine_tuning=True
        )

        # Evaluasi final
        from src.evaluate import evaluate_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_labels, all_preds = evaluate_model(model, test_loader, class_names, device)

        # Log confusion matrix sebagai artifact MLflow
        mlflow.log_artifact("confusion_matrix.png")

    print("\n🎉 SELURUH PROSES TRAINING SELESAI! 🎉")
    print("-> Jalankan: mlflow ui  (untuk melihat hasil di browser)")


if __name__ == '__main__':
    main()