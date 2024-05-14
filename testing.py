import hydra
import torch
import pandas as pd
import torchaudio
import os

from inference import inference
from src.sampler import Sampler78rpm
from denoise import (
    setup_network,
    setup_diff_params,
    setup_sampler,
    setup_data,
)

LOG_MESSAGE = """

Denoising {file_path} ...
Params:

INFER_XI - {infer_xi},
XI - {xi},
USE_GAIN - {use_gain},
USE_RANGE - {use_range},
POWER_WINDOW_SIZE - {power_window_size},
SIGMA_MIN - {sigma_min},
SIGMA_MAX - {sigma_max},
RHO - {rho},
CHECKPOINT - {checkpoint}

Results saved to {denoised_path}
"""


@hydra.main(config_path="conf", config_name="inference")
def main(args):
    torch.cuda.empty_cache()
    print("Preparing test files ...")
    info = get_test_info(args)
    model = setup_network(args)
    diff_params = setup_diff_params(args.diffusion_parameters)
    sampler = setup_sampler(model, diff_params, args)
    for _, row in info.iterrows():
        prepare_file(args, row, sampler)


def get_test_info(args: dict) -> pd.DataFrame:
    info_path = os.path.join(args.testing.folder, "sample-info.csv")
    return pd.read_csv(info_path)


def prepare_file(args: dict, row: pd.Series, sampler: Sampler78rpm):
    file = os.path.basename(row["arquivo"])
    panel_folder = os.path.join(args.testing.folder, f"painel-{args.testing.panel}")
    noisy_path = os.path.join(panel_folder, "noisy", file)
    denoised_path = os.path.join(panel_folder, args.testing.out, file)
    args.noisy_audio = noisy_path
    args.output_file = denoised_path
    args.inference.xi = get_xi(row[f"snr-painel-{args.testing.panel}"])
    args.inference.noise_gain = row[f"noise-gain-painel-{args.testing.panel}"]
    args.inference.snr_range = [0, 40]
    sampler.xi = args.inference.xi
    blocks, out_len = setup_data(args)
    print(get_log_message(args))
    denoised = inference(blocks, sampler, args, out_len)
    torchaudio.save(
        args.output_file,
        denoised,
        args.sample_rate,
        bits_per_sample=16,
        encoding="PCM_S",
    )


def get_xi(snr: float):
    # TODO code until 0.35 is just a stub for making more tests, remove when pushing final version
    if (snr >= 0) and (snr < 2):
        return 0.15
    if (snr >= 2) and (snr < 4):
        return 0.20
    if (snr >= 4) and (snr < 6):
        return 0.25
    if (snr >= 6) and (snr < 8):
        return 0.30
    if (snr >= 8) and (snr < 15):
        return 0.35
    if (snr >= 15) and (snr < 25):
        return 1.4
    if (snr >= 25) and (snr < 35):
        return 2.45
    if (snr >= 35) and (snr < 45):
        return 3.5
    raise ValueError("SNR Range Not Implemented")


def get_log_message(args: dict):
    fmt_dict = {
        "file_path": args.noisy_audio,
        "infer_xi": args.inference.infer_xi,
        "xi": args.inference.xi,
        "use_gain": args.inference.use_gain,
        "use_range": args.inference.use_range,
        "power_window_size": args.inference.denoising.power_window_size,
        "sigma_min": args.diffusion_parameters.sigma_min,
        "sigma_max": args.diffusion_parameters.sigma_max,
        "rho": args.diffusion_parameters.ro,
        "checkpoint": args.checkpoint,
        "denoised_path": args.output_file,
    }
    return LOG_MESSAGE.format(**fmt_dict)


if __name__ == "__main__":
    main()
