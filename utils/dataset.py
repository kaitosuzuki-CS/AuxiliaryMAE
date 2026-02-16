import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx]
        return data, label


def create_dataset(hps):
    def collate_fn(batch):
        data, labels = zip(*batch)

        data = torch.stack(data, dim=0)
        labels = torch.stack(labels, dim=0).long()

        return data, labels

    train_dataset = CustomDataset(hps.train_path)
    val_dataset = CustomDataset(hps.val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hps.train_bs,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hps.val_bs,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    return train_loader, val_loader
