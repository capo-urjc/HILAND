import copy

from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels.long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_targets(self):
        return copy.deepcopy(self.labels)
