# diffusion-audio-restoration

This repo is a fork of Eloi Moliner's [CQTDiff](https://github.com/eloimoliner/CQTdiff) diffusion model for solving inverse audio problems, [published in ICASSP 2023](https://arxiv.org/pdf/2210.15228).

For my master's, I adapted the sampling classes of the original model in order to remove background noise in 78 RPM historical recordings. I also retrained it (using the [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) dataset) for 44.1kHz sampling rate, as the original model was trained for 22.05kHz only. 

Although the model was only tested on 78 RPM noise removal tasks, it should work for any additive degradation (random or not).

## Dependencies

* [Python](https://www.python.org/downloads/) >= 3.9.1
* [pytorch](https://pytorch.org/get-started/) >= 1.11.0
* [torchaudio](https://pytorch.org/audio) >= 0.11.0
* [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 10.2
* [numpy](https://numpy.org/doc/stable/) 1.24.3

Other requirements can be seen in the ``requirements.txt`` file. This has project has also been tested with CUDA 11.8, pytorch 2.2.0, and torchaudio 2.2.0.

## Setup

First, make sure that CUDA and Python are properly installed. Then, clone the repository, create a virtual env, and install the packages listed in ``requirements.txt``. This can be done with the following commands.

```
git clone git@github.com:bvm810/diffusion-audio-restoration.git <destination-folder>
python -m venv <venv-name>
pip install --upgrade pip
pip install -r requirements.txt
```

If using pytorch higher than 1.11.0, verify that the correct numpy version is installed using ``pip freeze``. Newer numpy versions break Eloi Moliner's constant-Q transform implementation because of changes to ``np.clip``.

The model's weights should also be downloaded if not training from scratch. They can be downloaded [here](https://www02.smt.ufrj.br/~bernardo.miranda/master/weights/weights-2303999.pt).

For removing noise in 78 RPM recordings, a noise dataset is required. The one that was used in the dissertation is available (with metadata on the noise excerpts) [here](https://www02.smt.ufrj.br/~bernardo.miranda/master/weights/noise-dataset).

## Usage

### Inference on single noisy file

There are three ``.py``scripts that can be used for conditional sampling with diffusion in this repo: ``sample.py``, ``denoise.py``, and ``inference.py``. For the purpose of reproducing the work I developed in my master's, only ``inference.py`` should be used.

#### ``inference.py``

``inference.py`` is the inference file that was used for the first tests using diffusion to remove 78 RPM noise. It can work on four modes: fixed noise gain, fixed SNR range, estimated gain, and estimated SNR range (see Chapter 4 of my dissertation). In a practical setting, only the last two will be used.

This script is controlled by the ``inference.yaml`` configuration file. The ``audio_len`` attribute defines the block size used in overlap-and-add, and the full path to the model weights should be passed in the ``checkpoint``attribute.

General diffusion attributes can be set in ``diffusion_parameters`` section. Conditional sampling attributes (including $\xi'$) are handled in the ``inference`` section. Setting the ``use_gain`` and ``use_range`` flags boths to false denoises with fixed gain, and setting them both to true denoises with fixed SNR range. When using an estimated gain or an estimated SNR, the noise power estimate window size can be adjusted in the ``denoising`` subsection, which also has a field for a folder containing the noise inference set.

Below is an example of usage of the ``inference.py`` script
```
python inference.py noisy_audio=<input_path> output_file=<output_path> device="cuda:1" inference.xi=0.35 inference.use_gain=True inference.use_range=False
```

#### ``denoise.py``

``denoise.py`` is the first script I created to denoise a whole musical piece with a single command. It is controlled by the ``denoise.yaml`` configuration file, and is capable only of removing 78 RPM noise with a fixed noise gain or with a fixed SNR range. Aside from this, it is very similar to ``inference.py``. Gain and SNR range cannot be set simultaneously. 

#### ``sample.py``

``sample.py`` is the original script for inference by Eloi Moliner. I kept it mostly unchanged, except for the addition of a denoising mode. This script will apply the same degradation (78 RPM noise addition, low-pass filtering, clipping, etc) to all the ``.wav`` files in a given folder, and then it will use conditional diffusion to restore all the degraded versions. 

The behavior of ``sample.py`` is controlled by the ``conf.yaml`` configuration file. The sample rate of the ``.wav`` files should be the same that is set on the ``conf.yaml``'s ``sample_rate`` attribute. Similarly, the files should have lengths equal to the ``audio_len`` attribute. ``model_dir`` should have the path to the weigths folder, which must have the suffix given by ``checkpoint_id`` to be properly loaded. Diffusion inference parameters are controlled in the ``inference`` and ``diffusion_parameters``sections.

It should be noted that ``sample.py`` can only be used to restore small chunks of a musical recording. Therefore, if working on a complete musical piece, overlap-and-add needs to be done separately.

### Inference on a test folder

It is also possible to perform inference on a set of test signals using the ``testing.py`` script. This scripts reads a ``.csv`` file with information about the files being denoised, and restores them all using the same set of sampling parameters.

``testing.py`` expects to receive a parent directory path in the ``folder`` argument of the ``testing`` section of ``inference.yaml``. The parent directory should have the structure given below.
```bash
<parent-dir>
├── panel-1
│   ├── noisy
│   │   ├── Track 01.wav
│   │   ├── Track 02.wav
│   │   └── ...
│   ├── <out-folder>
│   └── ...
├── panel-2
│   ├── noisy
│   ├── <out-folder>
│   └── ...
├── ...
└── sample-info.csv
```

The idea is that test files can be divided into panels, so that multiple tests can be made with different sampling parameters. Each panel corresponds to one set of parameters. ``testing.py`` will receive a ``panel`` id number (set in the ``testing`` section of ``inference.yaml``). Then it will search for the tracks to be denoised in the ``panel-<id>/noisy`` folder. The names of the files to be denoised are indicated in the ``sample-info.csv`` file. 

``testing.py`` sets the value of $\xi'$ according to an estimate of the noisy signal SNR given by the user. Because of this, and because inference can be done with a fixed gain value, the ``sample-info.csv`` needs to have a ``snr-panel-<id>`` column and a ``noise-gain-panel-<id>`` column, so that each test file can have $\xi'$ and the gain adjusted properly. Below is an example of a valid ``sample-info.csv`` file 
```
arquivo,snr-painel-1,snr-painel-2,noise-gain-painel-1,noise-gain-painel-2
Track 01.wav,10,30,0.01,0.001
Track 02.wav,10,30,0.02,0.002
Track 03.wav,10,30,0.03,0.003
```

After denoising each file, the results will be saved in ``panel-<id>/<out-folder>``, with the same name of the original noisy file. ``out`` is also an argument of the ``testing`` section. The remaining parameters are defined in the same way as for inference for a single file. Below is an example of usage of the ``testing.py`` script
```
python testing.py testing.folder=<parent-dir> testing.panel=<id> testing.out=<out-folder> device="cuda:1" inference.use_gain=True inference.use_range=False
```

### Artificially degrading a clean recording

This ``noise-gain.py`` script in this repo can be used to artificially degrade a clean recording. It must be manually changed with the adequate input paths (for the noise sample and audio excerpt) and output paths.

## Changes with respect to CQTDiff

The main addition of this repo is the class ``Sampler78rpm`` in the ``sampler.py`` file of the source code. It inherits from the base ``Sampler`` class from CQTDiff, but performs conditional sampling to remove 78 RPM noise. This class needs to receive a path to a folder containing the noise inference set when it is instantiated. 

``Sampler78rpm`` has two conditional sampling methods: ``predict_gramophone_denoising(self, y_noisy, gain)``, and ``predict_gramophone_denoising_with_snr(self, y_noisy, range)``. The first uses an estimate of the noise gain during conditional sampling, and the latter uses a range of SNRs. 

The auxiliary functions for loading the noise samples, extending them if necessary, calculating the appropriate gain for a given SNR, and adding the noise samples to an intermediate diffusion sample can be found in ``src.util.denoising``.

## Experiment results

A detailed analysis of the results can be seen on my dissertations. For copyright reasons, I cannot publicly share the original validation and test files and their restored versions, but their metadata and individual results can be retrieved in the links below.

* Validation
    * [Metadata](https://www02.smt.ufrj.br/~bernardo.miranda/master/experiments/validation/validation-metadata.csv)
    * [Results](https://www02.smt.ufrj.br/~bernardo.miranda/master/experiments/validation/validation-results.csv)
* Tests with artificial signals
    * [Metadata](https://www02.smt.ufrj.br/~bernardo.miranda/master/experiments/artificial-signals/artificial-metadata.csv)
    * [Objective results](https://www02.smt.ufrj.br/~bernardo.miranda/master/experiments/artificial-signals/artificial-results.csv)
    * Subjective results (soon)
* Tests with real signals
    * Metadata(soon)
    * Results (soon)