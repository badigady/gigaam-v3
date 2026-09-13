import os
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/"

MODEL_FILES = {
    "encoder": "v3_e2e_rnnt_encoder.onnx",
    "decoder": "v3_e2e_rnnt_decoder.onnx",
    "joint": "v3_e2e_rnnt_joint.onnx",
    "vocab": "v3_e2e_rnnt_vocab.txt",
}


def default_models_dir():
    """Directory with the ONNX model files.

    Resolved from $GIGAAM_MODELS_DIR, otherwise ./models next to the project.
    """
    return Path(os.environ.get("GIGAAM_MODELS_DIR", "models"))


def required_paths(models_dir):
    models_dir = Path(models_dir)
    return {k: str(models_dir / name) for k, name in MODEL_FILES.items()}


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gigaam-v3/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                sys.stderr.write(
                    f"\r{dest.name}: {pct}% ({done // (1 << 20)} MiB / {total // (1 << 20)} MiB)"
                )
                sys.stderr.flush()
    sys.stderr.write("\n")
    os.replace(tmp, dest)


def ensure_models(models_dir=None):
    """Return paths to the four model files, downloading the missing ones.

    The model lives in the Hugging Face repo `istupakov/gigaam-v3-onnx`
    (the same weights used by the reference app): encoder (~885 MiB fp32),
    decoder, joint and vocab. Downloaded once into `models_dir`
    (default ./models, override with $GIGAAM_MODELS_DIR).
    """
    models_dir = Path(models_dir) if models_dir else default_models_dir()
    paths = required_paths(models_dir)
    missing = [k for k, p in paths.items() if not Path(p).exists()]
    if missing:
        sys.stderr.write(
            f"[gigaam] downloading missing model files from Hugging Face "
            f"(istupakov/gigaam-v3-onnx): {', '.join(missing)}\n"
        )
        for key in missing:
            _download(BASE_URL + MODEL_FILES[key], Path(paths[key]))
    return paths