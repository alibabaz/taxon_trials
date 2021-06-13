import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ['old_pad_seq', 'taxon_ds']

def old_pad_seq(seq):
    if(len(seq) < 4096):
        padded = torch.zeros(4096, dtype=torch.float)
        padded[:len(seq)] = seq
        return padded
    else:
        return seq
    
    
class taxon_ds(Dataset):
    def __init__(self, chunks, transform=None):
        self.chunks = chunks
        self.transform = transform
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        x = self.chunks[idx][1]
        if self.transform:
            x = self.transform(x)
        y = self.chunks[idx][2]
        x = x.to(torch.float)
        y = y.to(torch.long)
        return (x.unsqueeze(0), y)