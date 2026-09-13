import numpy as np
import onnxruntime as ort

from .features import compute_features, load_audio

BLANK = 1024
PRED_HIDDEN = 320
MAX_TOKENS_PER_STEP = 3

DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def load_vocab(path):
    entries = {}
    max_id = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            tok, _, id_ = line.strip().rpartition(" ")
            entries[int(id_)] = tok
            max_id = max(max_id, int(id_))
    assert max_id >= BLANK, f"vocab too small: max id {max_id} < blank {BLANK}"
    vocab = [""] * (max_id + 1)
    for id_, tok in entries.items():
        vocab[id_] = tok
    return vocab


class GigaAM:
    """GigaAM v3 e2e-RNNT transducer: audio bytes/path -> Russian text."""

    def __init__(self, encoder, decoder, joint, vocab_path, providers=None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0
        if providers is None:
            providers = DEFAULT_PROVIDERS
        self.encoder = ort.InferenceSession(encoder, sess_options=opts, providers=providers)
        self.decoder = ort.InferenceSession(decoder, sess_options=opts, providers=providers)
        self.joint = ort.InferenceSession(joint, sess_options=opts, providers=providers)
        self.providers = self.encoder.get_providers()
        self.vocab = load_vocab(vocab_path)

    def encode(self, feats, num_frames):
        out = self.encoder.run(None, {
            "audio_signal": feats[None, :, :num_frames].astype(np.float32),
            "length": np.array([num_frames], dtype=np.int64),
        })
        enc, enc_len = out
        enc_len = int(enc_len[0])
        enc_len = max(0, min(enc_len, enc.shape[2]))
        return enc[0, :, :enc_len].T.copy()  # [T', 768]

    def decode(self, enc_frames):
        h = np.zeros((1, 1, PRED_HIDDEN), dtype=np.float32)
        c = np.zeros((1, 1, PRED_HIDDEN), dtype=np.float32)
        tokens = []
        t_idx = 0
        emitted = 0
        while t_idx < enc_frames.shape[0]:
            last_token = tokens[-1] if tokens else BLANK
            dec_out = self.decoder.run(None, {
                "x": np.array([[last_token]], dtype=np.int64),
                "h.1": h,
                "c.1": c,
            })
            dec, h_new, c_new = dec_out
            joint_out = self.joint.run(None, {
                "enc": enc_frames[t_idx][None, :, None].astype(np.float32),
                "dec": np.ascontiguousarray(dec[0])[:, :, None].astype(np.float32),
            })
            token = int(np.argmax(joint_out[0]))
            if token != BLANK:
                h = h_new
                c = c_new
                tokens.append(token)
                emitted += 1
            if token == BLANK or emitted >= MAX_TOKENS_PER_STEP:
                t_idx += 1
                emitted = 0
        parts = [self.vocab[t] for t in tokens if t < len(self.vocab)]
        return "".join(parts).replace("▁", " ").strip()

    def transcribe_path(self, audio_path):
        waveform = load_audio(audio_path)
        feats, T = compute_features(waveform)
        if T == 0:
            return ""
        enc = self.encode(feats, T)
        return self.decode(enc)