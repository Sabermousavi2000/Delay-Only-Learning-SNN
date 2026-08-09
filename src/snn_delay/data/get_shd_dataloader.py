import torch
from torch.utils.data import DataLoader
from spikingjelly.datasets.shd import SpikingHeidelbergDigits


def get_shd_dataloader(data_dir, batch_size, time_steps=30, train=True):

    dataset = SpikingHeidelbergDigits(
        root=data_dir,
        train=train,
        data_type='frame',
        frames_number=time_steps,
        split_by='number'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=0,
        pin_memory=True
    )
    return dataloader
