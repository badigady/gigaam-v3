import os

import numpy as np

N_FFT = 320
HOP = 160
N_MELS = 64
N_FREQ = N_FFT // 2 + 1
CLAMP_MIN = 1e-9
CLAMP_MAX = 1e9

_ASSETS = os.path.join(os.path.dirname(__file__), "data", "gigaam_features_assets.npz")

_window = None
_fbank = None
_dcos = None
_dsin = None


def _assets():
    global _window, _fbank
    if _window is None:
        data = np.load(_ASSETS)
        _window = data["window"]
        _fbank = data["mel_fbank"].reshape(N_FREQ, N_MELS)
    return _window, _fbank


def _dft_tables():
    global _dcos, _dsin
    if _dcos is None:
        i = np.arange(N_FFT, dtype=np.float64)[None, :]
        f = np.arange(N_FREQ, dtype=np.float64)[:, None]
        ang = 2.0 * np.pi * f * i / N_FFT
        _dcos = np.cos(ang).astype(np.float32)
        _dsin = np.sin(ang).astype(np.float32)
    return _dcos, _dsin


def num_frames(n):
    if n < N_FFT:
        return 0
    return (n - N_FFT) // HOP + 1


def compute_features(waveform):
    """waveform: 1-D float32 numpy, 16 kHz mono.

    Returns (features, numFrames) with features of shape [N_MELS, numFrames].
    Exact port of onnx-asr's GigaamPreprocessorNumpy v3 (as used by the
    fbankAssets-loaded JS featurizer in the Type app).
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1-D")
    n = waveform.shape[0]
    T = num_frames(n)
    if T == 0:
        return np.empty((N_MELS, 0), dtype=np.float32), 0

    window, fbank = _assets()
    dcos, dsin = _dft_tables()

    frames = np.lib.stride_tricks.sliding_window_view(waveform[: n - N_FFT + 1], N_FFT)[::HOP]
    frames = frames * window

    power = frames @ dcos.T
    power = power * power + (frames @ dsin.T) * (frames @ dsin.T)

    mel = power @ fbank
    mel = np.clip(mel, CLAMP_MIN, CLAMP_MAX)
    mel = np.log(mel)
    return mel.T.astype(np.float32), T  # [N_MELS, T]


def load_audio(path, sample_rate=16000):
    import subprocess

    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-",
    ]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0