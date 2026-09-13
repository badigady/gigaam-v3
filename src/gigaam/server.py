import argparse
import io
import os
import sys
import tempfile
import time
import wave

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile

from . import models
from .features import compute_features, load_audio
from .model import DEFAULT_PROVIDERS, GigaAM

app = FastAPI(title="GigaAM v3 e2e-rnnt", version="1.0")
_model = None


def wav_to_waveform(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            frame_bytes = w.readframes(w.getnframes())
    except Exception:
        frame_bytes = None
        rate = channels = sampwidth = -1

    if rate == 16000 and channels == 1 and sampwidth == 2:
        pcm = np.frombuffer(frame_bytes, dtype=np.int16)
        return pcm.astype(np.float32) / 32768.0

    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return load_audio(path)
    finally:
        os.unlink(path)


def transcribe_wav(data):
    waveform = wav_to_waveform(data)
    feats, T = compute_features(waveform)
    if T == 0:
        return ""
    enc = _model.encode(feats, T)
    return _model.decode(enc)


@app.post("/v1/audio/transcriptions")
@app.post("/inference")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    t0 = time.time()
    text = transcribe_wav(data)
    dt = time.time() - t0
    sys.stderr.write(f"[gigaam_server] {file.filename}: {dt:.3f}s -> {text!r}\n")
    return {"text": text}


def main():
    ap = argparse.ArgumentParser(
        prog="gigaam-server",
        description="GigaAM v3 e2e-rnnt HTTP server (Vocalinux remote_api)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--models-dir", default=None,
                    help="directory with the ONNX model files (default: $GIGAAM_MODELS_DIR or ./models)")
    ap.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS,
                    help="ONNX Runtime execution providers")
    args = ap.parse_args()

    global _model
    paths = models.ensure_models(args.models_dir or models.default_models_dir())
    t0 = time.time()
    _model = GigaAM(paths["encoder"], paths["decoder"], paths["joint"], paths["vocab"],
                    providers=args.providers)
    sys.stderr.write(f"loaded model in {time.time() - t0:.2f}s, providers: {_model.providers}\n")
    sys.stderr.write(f"listening on http://{args.host}:{args.port} "
                     f"(POST /v1/audio/transcriptions, POST /inference)\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()