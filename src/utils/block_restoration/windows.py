import torch


def window_with_fade(window_size: int, fade_size: int) -> torch.Tensor:
    return complementary_flat_window_with_fade(window_size, fade_size) ** 2


def complementary_flat_window_with_fade(
    window_size: int, fade_size: int
) -> torch.Tensor:
    kernel = torch.hann_window(fade_size)
    fade_in = torch.sqrt(torch.cumsum(kernel, dim=0) / torch.sum(kernel, dim=0))
    flat = torch.ones(window_size - 2 * fade_size)
    fade_out = torch.flip(fade_in, dims=[0])
    return torch.cat((fade_in, flat, fade_out)).unsqueeze(0)


def flat_window(size: int) -> torch.Tensor:
    return torch.ones((1, size))
