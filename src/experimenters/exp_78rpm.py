import os
import torch
import random
from src.sampler import Sampler78rpm
from src.experimenters.exp_base import Exp_Base
import src.utils.bandwidth_extension as utils_bwe
import src.utils.denoising as utils_denoising
import src.utils.logging as utils_logging


class Exp_Denoising(Exp_Base):
    def __init__(self, args, plot_animation):
        super().__init__(args)
        self.__plot_animation = plot_animation
        self.sampler = Sampler78rpm(
            self.model,
            self.diff_parameters,
            self.args,
            args.inference.xi,
            order=2,
            data_consistency=args.inference.data_consistency,
            rid=self.__plot_animation,
        )

        self.__filter_final = utils_bwe.get_FIR_lowpass(
            100, self.args.sample_rate // 2 - 500, 1, self.args.sample_rate
        )
        self.__filter_final = self.__filter_final.to(self.device)

        # noise file used for the experiment, needs to be the same for all chunks
        self.choose_noise(self.args.inference.denoising.noise_files_path)

        # denoising specific
        n = "78rpm_noisy" + "_" + str(self.args.inference.denoising.noise_gain)
        self.path_degraded = os.path.join(
            self.path_sampling, n
        )  # path for the noisy outputs

        # ensure the path exists
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)

        self.path_original = os.path.join(
            self.path_sampling, "original"
        )  # this will need a better organization
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)

        # path where the output will be saved
        n = "78rpm_denoised" + "_" + str(self.args.inference.denoising.noise_gain)
        self.path_reconstructed = os.path.join(
            self.path_sampling, n
        )  # path for the denoised outputs
        # ensure the path exists
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)

    def choose_noise(self, noise_folder: str):
        # files = os.listdir(noise_folder)
        # noise_file = random.choice(files)
        # self.noise_path = os.path.join(noise_folder, noise_file)
        self.noise_path = self.args.inference.denoising.noise_file

    def conduct_experiment(self, seg, name):
        # assuming now that the input has the expected shape
        print(seg.shape[-1])
        assert seg.shape[-1] == self.args.audio_len

        seg = seg.to(self.device)
        audio_path = utils_logging.write_audio_file(
            seg, self.args.sample_rate, name, self.path_original + "/"
        )
        print(audio_path)

        crossfade_duration = self.args.inference.denoising.crossfade_duration
        noise = utils_denoising.get_noise(
            seg.shape[-1], crossfade_duration, self.noise_path
        )
        noise = noise.to(self.device)

        # apply noise
        gain = self.args.inference.denoising.noise_gain
        y = utils_denoising.add_noise_with_gain(seg, noise, gain)

        # save noisy audio file
        audio_path_clipped = utils_logging.write_audio_file(
            y, self.args.sample_rate, name, self.path_degraded + "/"
        )
        print(audio_path)

        # input("stop")
        if self.__plot_animation:
            x_hat, data_denoised, t = self.sampler.predict_gramophone_denoising(y, gain)
            fig = utils_logging.diffusion_CQT_animation(
                self.path_reconstructed,
                data_denoised,
                t,
                self.args.stft,
                name="animation" + name,
                compression_factor=8,
            )
        else:
            x_hat = self.sampler.predict_gramophone_denoising(y, gain)

        # apply low pass filter to remove annoying artifacts at the nyquist frequency. I should try to fix this issue in future work
        x_hat = utils_bwe.apply_low_pass(x_hat, self.__filter_final, "firwin")

        # save reconstructed audio file
        audio_path_result = utils_logging.write_audio_file(
            x_hat, self.args.sample_rate, name, self.path_reconstructed + "/"
        )
        print(audio_path)
        if self.__plot_animation:
            return audio_path_clipped, audio_path_result, fig
        else:
            return audio_path_clipped, audio_path_result
