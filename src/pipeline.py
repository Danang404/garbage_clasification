from .split_data import split_data_physically
from .data_loader import get_dataloaders

class GarbagePipeline:
    def execute(self):
        # Tahap 2 (Splitting)
        split_data_physically()
        
        # Tahap 3 (Dataloader)
        train_ds, val_ds, test_ds = get_dataloaders()
        
        print("\nPIPELINE PREPROCESSING SELESAI: Data siap untuk Training Model")
        return train_ds, val_ds, test_ds