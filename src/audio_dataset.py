import torch
import torchaudio
from torch.utils.data import Dataset
from src.utils.block_restoration.windows import flat_window
from src.utils.block_restoration.analysis import divide_signal


class SingleAudioDataset(Dataset):
    def __init__(self, audio: torch.Tensor, args: dict):
        self.audio = audio
        self.block_duration = args.block_duration
        self.overlap_duration = args.overlap_duration
        self.fs = args.sample_rate
        self._divide_signal()

    def _divide_signal(self):
        superposition = int(self.overlap_duration * self.fs)
        block_size = int(self.block_duration * self.fs)
        window = flat_window(block_size)
        self.blocks = divide_signal(self.audio, window, superposition)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]
