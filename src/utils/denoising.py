import torchaudio
import torch
import os


def extend(audio: torch.Tensor, crossfade_size: int) -> torch.Tensor:
    """Loops audio tensor with itself with crossfade. Returns tensor of size (n_samples,)"""
    window = torch.hann_window(crossfade_size * 2)
    extended_audio = torch.zeros((2 * audio.shape[-1] - crossfade_size))
    extended_audio[: (audio.shape[-1] - crossfade_size)] = audio[:-crossfade_size]
    extended_audio[(audio.shape[-1] - crossfade_size) : audio.shape[-1]] += (
        audio[-crossfade_size:] * window[crossfade_size:]
    )
    extended_audio[(audio.shape[-1] - crossfade_size) : audio.shape[-1]] += (
        audio[:crossfade_size] * window[:crossfade_size]
    )
    extended_audio[audio.shape[-1] :] = audio[crossfade_size:]
    return extended_audio


def calculate_period_size(fs: float, rpm: float = 78.0) -> int:
    return int(60 / rpm * fs)


def separate_full_periods(
    audio: torch.Tensor, period_size: int, crossfade_size: int
) -> torch.Tensor:
    leftover_samples = audio.shape[-1] % period_size
    if leftover_samples < crossfade_size:
        return audio[: -(leftover_samples + period_size - crossfade_size)]
    return audio[: -(leftover_samples - crossfade_size)]


def get_noise(n_samples: int, crossfade_time: float, noise_path: str) -> torch.Tensor:
    """Gets noise of len n_samples by looping the noise file given. Returns tensor of size (n_samples,)"""
    noise, fs = torchaudio.load(noise_path)
    crossfade_size = int(crossfade_time * fs)
    noise = torch.mean(noise, dim=0)  # convert to mono
    period_size = calculate_period_size(fs)
    noise = separate_full_periods(noise, period_size, crossfade_size)
    while noise.shape[-1] < n_samples:
        noise = extend(noise, crossfade_size)
    return noise[:n_samples]


def get_noise_gain_for_snr(
    audio: torch.Tensor, noise: torch.Tensor, snr: float
) -> float:
    audio_energy = torch.sum(torch.pow(audio, 2))
    noise_energy = torch.sum(torch.pow(noise, 2))
    # snr = 10 log (Ps/Pn) --> G = sqrt((10^(snr/10))^(-1) * Es/En)
    gain = torch.sqrt((1 / 10 ** (snr / 10)) * (audio_energy / noise_energy))
    return gain


def add_noise_with_snr(
    audio: torch.Tensor, noise: torch.Tensor, snr: float
) -> torch.Tensor:
    """Adds noise to signal with given SNR. Expects two tensors of size (n_samples,). Returns tensor of same shape"""
    audio = audio / 2
    gain = get_noise_gain_for_snr(audio, noise, snr)
    return audio + gain * noise


def add_noise_with_gain(
    audio: torch.Tensor, noise: torch.Tensor, gain: float
) -> torch.Tensor:
    return (audio + gain * noise) / 2


if __name__ == "__main__":
    # tests, as there is no test suite in moliner's original code

    # crossfade test
    # IN_PATH = "/nfs/home/bernardo.miranda/Documents/pesquisa/datasets/base-ruidos-gramofone/all/78__sandy-macfarlane_gbia0006642_segment-01.wav"
    # OUT_PATH = "/nfs/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/denoising/extension-test.wav"
    # audio, fs = torchaudio.load(IN_PATH)
    # audio = torch.sum(audio, dim=0) / 2
    # crossfade_size = int(0.5 * fs)
    # extended = extend(audio, crossfade_size)
    # extended = extended.unsqueeze(0)
    # torchaudio.save(OUT_PATH, extended, fs)

    # noise test
    IN_PATH = "/nfs/home/bernardo.miranda/Documents/pesquisa/datasets/base-ruidos-gramofone/all/78__sandy-macfarlane_gbia0006642_segment-01.wav"
    OUT_PATH = "/nfs/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/denoising/extension-test-2.wav"
    noise_size = 10 * 44100
    noise = get_noise(noise_size, 0.5, IN_PATH)
    noise = noise.unsqueeze(0)
    torchaudio.save(OUT_PATH, noise, 44100)

    # audio with noise and specified SNR test
    SNR = 10
    AUDIO_PATH = "/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/declipping/original/AC-FlauteandoNaChacrinha-flauta-f.wav"
    NOISE_PATH = "/nfs/home/bernardo.miranda/Documents/pesquisa/datasets/base-ruidos-gramofone/all/78__sandy-macfarlane_gbia0006642_segment-01.wav"
    OUT_PATH = f"/nfs/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/denoising/extension-test-3_SNR-{SNR}.wav"
    audio, fs = torchaudio.load(AUDIO_PATH)
    audio = torch.sum(audio, dim=0) / 2
    noise = get_noise(audio.shape[-1], 0.5, NOISE_PATH)
    gain = get_noise_gain_for_snr(audio / 2, noise, SNR)
    noisy_audio = add_noise_with_snr(audio, noise, SNR)
    torchaudio.save(OUT_PATH, noisy_audio.unsqueeze(0), fs)
    assert torch.allclose(noisy_audio - gain * noise, audio / 2)
    audio_power = torch.sum(torch.pow(noisy_audio - gain * noise, 2))
    noise_power = torch.sum(torch.pow(gain * noise, 2))
    calculated_snr = 10 * torch.log10(audio_power / noise_power)
    assert abs(SNR - calculated_snr) < 1e-8
    assert abs(SNR - calculated_snr) / SNR < 1e-4
