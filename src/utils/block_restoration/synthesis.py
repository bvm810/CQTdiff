import torchaudio
import torch
import os
from typing import List


def overlapp_and_add(
    blocks: torch.Tensor, window: torch.Tensor, superposition: int
) -> torch.Tensor:
    blocks = blocks * window.unsqueeze(0)
    num_blocks, num_channels, block_size = blocks.shape
    hop = blocks.shape[-1] - superposition
    output_len = block_size + hop * (num_blocks - 1)
    out = torch.zeros((num_channels, output_len))
    for idx in range(num_blocks):
        start = idx * hop
        end = start + block_size
        out[:, start:end] += blocks[idx, :, :]
    return out


def is_wave(file: str) -> bool:
    return os.path.splitext(file)[-1] == ".wav"


def load_blocks(in_folder: str) -> torch.Tensor:
    # assumes all files in in_folder have the same size
    paths = [os.path.join(in_folder, name) for name in os.listdir(in_folder)]
    wavs = [path for path in paths if is_wave(path)]
    block_tensors = [torchaudio.load(wav)[0].unsqueeze(0) for wav in wavs]
    return torch.cat(block_tensors, dim=0)
