import torch
import torchaudio
from typing import List


def save_blocks(blocks: torch.Tensor, fs: float, out_paths: List[str]):
    for i in range(blocks.shape[0]):
        torchaudio.save(out_paths[i], blocks[i, :, :], fs)


def calculate_window_padding_size(
    signal_size: int, block_size: int, superposition: int
) -> int:
    hop = block_size - superposition
    extra_samples = (signal_size - block_size) % hop
    pad_size = hop - extra_samples
    return int(pad_size)


def pad_signal(
    signal: torch.Tensor, block_size: int, superposition: int
) -> torch.Tensor:
    superposition_pad = torch.zeros(signal.shape[0], superposition)
    signal = torch.cat((superposition_pad, signal, superposition_pad), dim=1)
    pad_size = calculate_window_padding_size(
        signal.shape[-1], block_size, superposition
    )
    left_padding = torch.zeros((signal.shape[0], pad_size // 2))
    right_padding = torch.zeros((signal.shape[0], pad_size // 2 + pad_size % 2))
    return torch.cat((left_padding, signal, right_padding), dim=1)


def divide_signal(
    signal: torch.Tensor, window: torch.Tensor, superposition: int
) -> List[torch.Tensor]:
    block_size = window.shape[-1]
    hop = block_size - superposition
    padded_signal = pad_signal(signal, block_size, superposition)
    blocks = padded_signal.unfold(1, block_size, hop)
    windowed_blocks = blocks * window.unsqueeze(0)  # (n_channels, n_blocks, block_size)
    return torch.movedim(windowed_blocks, 1, 0)  # (n_blocks, n_channels, block_size)
