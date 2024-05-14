import hydra
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from scipy.signal import firls
from scipy.io import savemat
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from src.models.unet_cqt import Unet_CQT
from src.utils.setup import load_ema_weights
from src.audio_dataset import SingleAudioDataset
from src.utils.block_restoration.windows import window_with_fade
from src.utils.block_restoration.synthesis import overlapp_and_add
from src.sampler import Sampler78rpm
from src.sde import VE_Sde_Elucidating


@hydra.main(config_path="conf", config_name="denoise")
def main(args):
    torch.cuda.empty_cache()
    print("Starting run ...")
    print(f"Noisy file: {args.noisy_audio}")
    print(f"Output file: {args.output_file}")
    print(f"Device: {args.device}")
    print(f"Noise Gain: {args.inference.denoising.noise_gain}")
    print(f"SNR Range: {args.inference.denoising.snr_range}")
    print(f"Xi: {args.inference.xi}\n\n\n")
    model = setup_network(args)
    diff_params = setup_diff_params(args.diffusion_parameters)
    sampler = setup_sampler(model, diff_params, args)
    prefilter, postfilter = setup_filters(args.filtering, args.sample_rate)
    blocks, out_len = setup_data(args, prefilter)
    denoised = denoise(blocks, sampler, args, out_len, postfilter)
    torchaudio.save(
        args.output_file,
        denoised,
        args.sample_rate,
        bits_per_sample=16,
        encoding="PCM_S",
    )


def setup_network(args: dict) -> torch.nn.Module:
    model = Unet_CQT(args, args.device)
    print(f"Loading from {args.checkpoint}")
    model = load_ema_weights(model, args.checkpoint)
    return model


def setup_diff_params(diff_args: dict) -> VE_Sde_Elucidating:
    return VE_Sde_Elucidating(diff_args, diff_args.sigma_data)


def setup_sampler(
    model: Unet_CQT, diff_params: VE_Sde_Elucidating, args: dict
) -> Sampler78rpm:
    return Sampler78rpm(
        model=model,
        diff_params=diff_params,
        args=args,
        xi=args.inference.xi,
        order=args.inference.order,
        data_consistency=args.inference.data_consistency,
        rid=False,
    )


def setup_filters(
    filter_args: dict, fs: float
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    freqs = get_filter_freqs(fs, filter_args["points_per_band"])
    gains = get_filter_gains(
        filter_args["points_per_band"], filter_args["bandpass_power"]
    )
    inverse_gains = 1 / gains
    prefilter, postfilter = None, None
    if filter_args["use_prefilter"]:
        prefilter = firls(filter_args["order"], freqs, gains, fs=fs)
        prefilter = torch.Tensor(prefilter)
    if filter_args["use_postfilter"]:
        postfilter = firls(filter_args["order"], freqs, inverse_gains, fs=fs)
        postfilter = torch.Tensor(postfilter)
    if filter_args["save_path"] is not None:
        matfile = {
            "prefilter": prefilter if prefilter is not None else [],
            "postfilter": postfilter if postfilter is not None else [],
        }
        savemat(filter_args["save_path"], matfile)
    return prefilter, postfilter


def get_filter_freqs(fs: float, points_per_band: int):
    reject_end = (fs / 4) * 0.9
    pass_start = (fs / 4) * 1.1
    reject_band = np.linspace(0, reject_end, num=points_per_band)
    transition_band = np.linspace(reject_end, pass_start, num=points_per_band)
    pass_band = np.linspace(pass_start, fs / 2, num=points_per_band)
    return np.concatenate((reject_band, transition_band, pass_band))


def get_filter_gains(points_per_band: int, power_gain: float):
    reject_band = np.ones(points_per_band)
    pass_band = (power_gain ** (1 / 2)) * np.ones(points_per_band)
    transition_band = np.sqrt(np.linspace(1, power_gain, points_per_band))
    return np.concatenate((reject_band, transition_band, pass_band))


def setup_data(args: dict, prefilter: Optional[torch.Tensor] = None) -> DataLoader:
    audio = load_audio(args.noisy_audio, args.sample_rate)
    if prefilter is not None:
        audio = filter_signal(audio, prefilter)
    data = SingleAudioDataset(audio, args)
    loader = DataLoader(data, batch_size=1, shuffle=False)
    return loader, audio.shape[-1]


def filter_signal(signal: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
    denominator = torch.zeros(fir.shape)
    denominator[0] = 1
    return torchaudio.functional.filtfilt(signal, denominator, fir, clamp=False)


def load_audio(file: str, rate: float):
    audio, fs = torchaudio.load(file)
    if fs != rate:
        raise ValueError(f"Sampling rate {fs} of the file is not {rate}")
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    return audio


def denoise(
    blocks: DataLoader,
    sampler: Sampler78rpm,
    args: dict,
    out_len: int,
    postfilter: Optional[torch.Tensor],
):
    sampler.model.to(args.device)
    denoised_blocks = torch.Tensor().to(args.device)
    gain = args.inference.denoising.noise_gain
    snr_range = args.inference.denoising.snr_range
    if gain is not None and snr_range is not None:
        raise ValueError("Cannot have SNR range and noise gain set simultaneously")
    for block in tqdm(blocks):
        b = block[0].to(args.device)
        if gain is not None:
            denoised_block = sampler.predict_gramophone_denoising(b, gain)
        if snr_range is not None:
            denoised_block = sampler.predict_gramophone_denoising_with_snr(b, snr_range)
        denoised_block = denoised_block.unsqueeze(0)
        denoised_blocks = torch.cat((denoised_blocks, denoised_block), dim=0)
    denoised_blocks = denoised_blocks.to("cpu")
    reconstructed = reconstruct(denoised_blocks, out_len, args)
    if postfilter is not None:
        reconstructed = filter_signal(reconstructed, postfilter)
    return reconstructed


def reconstruct(blocks: torch.Tensor, audio_len: int, args: dict) -> torch.Tensor:
    block_size = int(args.block_duration * args.sample_rate)
    superposition = int(args.overlap_duration * args.sample_rate)
    window = window_with_fade(block_size, superposition)
    reconstructed = overlapp_and_add(blocks, window, superposition)
    pad_size = reconstructed.shape[-1] - audio_len
    left_pad_size = pad_size // 2
    right_pad_size = pad_size // 2 + pad_size % 2
    reconstructed = reconstructed[:, left_pad_size:-(right_pad_size)]
    return reconstructed


if __name__ == "__main__":
    main()
