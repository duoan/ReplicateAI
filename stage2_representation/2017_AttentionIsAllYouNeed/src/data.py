import torch


class ToyDataset(torch.utils.data.Dataset):

    def __init__(self, vocab_size=20, seq_len=10, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.samples = torch.randint(1, vocab_size, (num_samples, seq_len))


    def __getitem__(self, index):
        x = self.samples[index]
        y = torch.cat([x[1:], torch.tensor([0])])  # shift right for teacher forcing
        return x, y

    def __len__(self):
        return len(self.samples)