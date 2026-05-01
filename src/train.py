import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train_model(model, train_loader, val_loader, class_weights_dict, num_classes, epochs, learning_rate, save_name, is_fine_tuning=False):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Menggunakan hardware komputasi: {device.type.upper()}")
    model = model.to(device)

    weights_tensor = torch.tensor(
        [class_weights_dict[i] for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.2, patience=2, verbose=True
    )
    scaler = GradScaler()

    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 3 if is_fine_tuning else 5
    os.makedirs('models', exist_ok=True)

    phase_name = "fine_tuning" if is_fine_tuning else "initial_transfer"

    with mlflow.start_run(run_name=phase_name, nested=True):
        # Log hyperparameter
        mlflow.log_params({
            "phase": phase_name,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "num_classes": num_classes,
            "patience_limit": patience_limit,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
        })

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            # --- TRAIN PHASE ---
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            train_bar = tqdm(train_loader, desc="Training", leave=False)

            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct.double() / total

            # --- VALIDATION PHASE ---
            model.eval()
            val_running_loss, val_correct, val_total = 0.0, 0, 0
            val_bar = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels.data)
                    val_total += labels.size(0)

            val_loss = val_running_loss / val_total
            val_acc = val_correct.double() / val_total

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # Log metrics per epoch ke MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": float(train_acc),
                "val_loss": val_loss,
                "val_acc": float(val_acc),
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            scheduler.step(val_acc)

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'models/{save_name}.pth')
                print(f"Model membaik! Tersimpan sebagai {save_name}.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Model tidak membaik. Kesempatan tersisa: {patience_limit - patience_counter}")
                if patience_counter >= patience_limit:
                    print("Early Stopping terpicu! Menghentikan proses ini.")
                    break

        # Log best val acc dan simpan model ke MLflow
        mlflow.log_metric("best_val_acc", float(best_val_acc))
        mlflow.pytorch.log_model(model, artifact_path=save_name)
        print(f"-> Model dan metrics berhasil dicatat di MLflow!")

    model.load_state_dict(torch.load(
        f'models/{save_name}.pth',
        map_location=device,
        weights_only=True
    ))
    return model