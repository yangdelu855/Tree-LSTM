from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ids):
        self.ids = ids

    def __getitem__(self, item):
        return self.ids[item]

    def __len__(self):
        return len(self.ids)
