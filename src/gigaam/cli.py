import argparse
import sys
import time
from pathlib import Path

from . import models
from .model import DEFAULT_PROVIDERS, GigaAM


def main():
    ap = argparse.ArgumentParser(
        prog="gigaam",
        description="GigaAM v3 e2e-rnnt transcription (ONNX Runtime)",
    )
    ap.add_argument("audio", nargs="+", help="audio file(s) to transcribe (any ffmpeg input)")
    ap.add_argument("--models-dir", default=None,
                    help="directory with the ONNX model files (default: $GIGAAM_MODELS_DIR or ./models)")
    ap.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS,
                    help="ONNX Runtime execution providers")
    ap.add_argument("--time", action="store_true", help="print per-file timing")
    args = ap.parse_args()

    paths = models.ensure_models(args.models_dir or models.default_models_dir())
    model = GigaAM(paths["encoder"], paths["decoder"], paths["joint"], paths["vocab"],
                   providers=args.providers)
    print(f"providers: {model.providers}, vocab size: {len(model.vocab)}", file=sys.stderr)

    for path in args.audio:
        t0 = time.time()
        text = model.transcribe_path(path)
        dt = time.time() - t0
        if args.time:
            print(f"[{dt:.2f}s] {path}" + " " * max(0, 30 - len(path)), file=sys.stderr)
        print(text)


if __name__ == "__main__":
    main()