import torch
import torchaudio
from src.utils.denoising import get_noise, get_noise_gain_for_snr, add_noise_with_snr

if __name__ == "__main__":
    PRELUDE = 7
    NOISE_PATH = "/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/44100/denoising/noises/original-78_1-olympia-hippodrome-2-go_ringling-bros-and-barnum-baileys-band-alexander-hu_gbia0263963b_segment-01.wav"
    AUDIO_PATH = f"/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/denoising-historical-recordings/experiments/Chopin Op 28 Prelude {PRELUDE} - C. Katsaris.wav"
    CROSSFADE_TIME = 0.1
    SNR = 40

    audio, fs = torchaudio.load(AUDIO_PATH)
    # convert audio to mono
    audio = torch.mean(audio, dim=0)
    noise = get_noise(audio.shape[-1], CROSSFADE_TIME, NOISE_PATH)
    # normalize power
    noise = noise / torch.sqrt(torch.mean(torch.pow(noise, 2)))
    gain = get_noise_gain_for_snr(audio, noise, SNR)
    noisy_file = add_noise_with_snr(audio, noise, SNR)

    print(f"Audio: {AUDIO_PATH}")
    print(f"Noise: {NOISE_PATH}")
    print(f"SNR {SNR}")
    print(f"Gain: {gain}")

    OUT_PATH = f"/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/44100/denoising/SNR_{SNR}dB/prelude-{PRELUDE}-original.wav"
    torchaudio.save(OUT_PATH, audio.unsqueeze(0), fs)
    OUT_PATH = f"/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/44100/denoising/SNR_{SNR}dB/prelude-{PRELUDE}-noisy.wav"
    torchaudio.save(OUT_PATH, noisy_file.unsqueeze(0), fs)
    OUT_PATH = f"/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/44100/denoising/SNR_{SNR}dB/prelude-{PRELUDE}-noisy-double-amp.wav"
    torchaudio.save(OUT_PATH, 2 * noisy_file.unsqueeze(0), fs)
    OUT_PATH = f"/home/bernardo.miranda/Documents/pesquisa/dissertacao/artigos-moliner/cqt-diff/experiments/44100/denoising/SNR_{SNR}dB/prelude-{PRELUDE}-half-amplitude.wav"
    torchaudio.save(OUT_PATH, (audio / 2).unsqueeze(0), fs)
