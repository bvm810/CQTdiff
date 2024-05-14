import hydra
import torch
import math
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.functional import conv1d

from src.sampler import Sampler78rpm
from denoise import (
    reconstruct,
    setup_network,
    setup_diff_params,
    setup_sampler,
    setup_data,
)

XI_REGRESSION_INTERCEPT = -4.0884532164
XI_REGRESSION_SLOPE = -1.0245766357


@hydra.main(config_path="conf", config_name="inference")
def main(args):
    torch.cuda.empty_cache()
    print("Starting run ...")
    print(f"Noisy file: {args.noisy_audio}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"Noise floor window size: {args.inference.denoising.power_window_size}")
    print(f"Xi: {args.inference.xi}\n\n\n")
    model = setup_network(args)
    diff_params = setup_diff_params(args.diffusion_parameters)
    sampler = setup_sampler(model, diff_params, args)
    blocks, out_len = setup_data(args)
    denoised = inference(blocks, sampler, args, out_len)
    torchaudio.save(
        args.output_file,
        denoised,
        args.sample_rate,
        bits_per_sample=16,
        encoding="PCM_S",
    )


def inference(blocks: DataLoader, sampler: Sampler78rpm, args: dict, out_len: int):
    sampler.model.to(args.device)
    denoised_blocks = torch.Tensor().to(args.device)
    window_size = args.inference.denoising.power_window_size
    floor, ceil = get_floor_and_ceil_powers(args)
    if args.inference.infer_xi == True:
        sampler.xi = get_xi_estimate(floor)
        print(f"XI Estimate: {sampler.xi}")
    if args.inference.use_gain == True:
        print(f"Estimated gain: {math.sqrt(floor)}\n")
    for block in tqdm(blocks):
        b = block[0].to(args.device)
        if (args.inference.use_range == True) and (args.inference.use_gain == True):
            print(f"Denoising with fixed SNR Range {args.inference.snr_range}")
            denoised_block = sampler.predict_gramophone_denoising_with_snr(
                b, args.inference.snr_range
            )
        if (args.inference.use_range == False) and (args.inference.use_gain == False):
            print(f"Denoising with fixed gain {args.inference.noise_gain}")
            denoised_block = sampler.predict_gramophone_denoising(
                b, args.inference.noise_gain
            )
        if (args.inference.use_range == True) and (args.inference.use_gain == False):
            print("Denoising estimating SNR range")
            range_min = get_snr_range_min(b, floor, window_size)
            range_width = get_snr_range_width(floor, ceil)
            snr_range = [range_min, range_min + range_width]
            denoised_block = sampler.predict_gramophone_denoising_with_snr(b, snr_range)
        if (args.inference.use_range == False) and (args.inference.use_gain == True):
            print("Denoising estimating noise gain")
            denoised_block = sampler.predict_gramophone_denoising(b, math.sqrt(floor))
        denoised_block = denoised_block.unsqueeze(0)
        denoised_blocks = torch.cat((denoised_blocks, denoised_block), dim=0)
    denoised_blocks = denoised_blocks.to("cpu")
    reconstructed = reconstruct(denoised_blocks, out_len, args)
    return reconstructed


def get_floor_and_ceil_powers(args: dict):
    print("\nGetting maximum and minimal signal powers ...")
    signal, _ = torchaudio.load(args.noisy_audio)
    powers = get_powers(signal, args.inference.denoising.power_window_size)
    print("\n")
    return torch.min(powers).item(), torch.max(powers).item()


def get_snr_range_min(block: torch.Tensor, floor: float, window_size: int):
    block_powers = get_powers(block, window_size)
    ratio = (torch.max(block_powers).item() - floor) / (max(floor, 1e-15))
    return 10 * math.log10(ratio)


def get_snr_range_width(floor: float, ceil: float):
    reduction = 10 * math.log10(max(floor, 1e-12) / ceil)
    return 43 + max(-40, reduction)


def get_xi_estimate(floor: float):
    return max(0.35, XI_REGRESSION_SLOPE * math.log10(floor) + XI_REGRESSION_INTERCEPT)


def get_powers(signal: torch.Tensor, window_size: int):
    kernel = torch.ones(window_size) / window_size
    kernel = kernel.view(1, 1, -1).to(signal.device)
    squared_signal = (torch.pow(signal, 2)).view(1, signal.shape[0], -1)
    powers = conv1d(squared_signal, kernel)
    return powers.squeeze()


if __name__ == "__main__":
    main()
