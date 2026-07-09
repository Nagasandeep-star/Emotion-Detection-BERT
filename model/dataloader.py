from torch.utils.data import DataLoader


def create_dataloader(dataset, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader.

    Args:
        dataset: EmotionDataset object
        batch_size: Number of samples per batch
        shuffle: Shuffle data during training

    Returns:
        DataLoader object
    """

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader