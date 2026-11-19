"""VITS model loader + synthesis (project-local weights, GPU, no HF caches)."""

from __future__ import annotations

import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

DEFAULT_CKPT = Path("models/vits/G_100000.pth")
DEFAULT_CONFIG = Path("third_party/vits/configs/nepali_base.json")


class VitsEngine:
    def __init__(self, vits_dir: Path, checkpoint: Path,
                 device: str = "cuda", noise_scale: float = 0.667,
                 noise_scale_w: float = 0.8, length_scale: float = 1.0):
        self.vits_dir = Path(vits_dir)
        self.checkpoint = Path(checkpoint)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.noise_scale = noise_scale
        self.noise_scale_w = noise_scale_w
        self.length_scale = length_scale
        self._net = None
        self._hps = None
        self._symbols = None

    def load(self) -> None:
        if self._net is not None:
            return
        if str(self.vits_dir) not in sys.path:
            sys.path.insert(0, str(self.vits_dir))
        from text import text_to_sequence, symbols
        from models import SynthesizerTrn
        import utils

        hps = utils.get_hparams_from_file(str(self.vits_dir / "configs" /
                                              "nepali_base.json"))
        net = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        net.eval()
        utils.load_checkpoint(str(self.checkpoint), net, None)
        self._net = net.to(self.device)
        self._hps = hps
        self._symbols = symbols
        self._text_to_sequence = text_to_sequence
        import commons
        self._commons = commons

    @property
    def sample_rate(self) -> int:
        self.load()
        return self._hps.data.sampling_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Nepali text -> mono float32 waveform in [-1, 1]."""
        self.load()
        text = unicodedata.normalize("NFC", text)
        seq = self._text_to_sequence(text, self._hps.data.text_cleaners)
        if not seq:
            raise ValueError("empty text sequence after cleaning")
        seq = self._commons.intersperse(seq, 0)  # REQUIRED by vits inference
        x = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        x_len = torch.LongTensor([len(seq)]).to(self.device)
        t0 = time.time()
        with torch.no_grad():
            audio = self._net.infer(
                x, x_len,
                noise_scale=self.noise_scale,
                noise_scale_w=self.noise_scale_w,
                length_scale=self.length_scale,
            )[0][0, 0].cpu().numpy().astype(np.float32)
        self.last_seconds = time.time() - t0
        return audio

    def free(self) -> None:
        """Drop the model from VRAM (for a clean hand-off/exit)."""
        if self._net is not None:
            self._net = self._net.cpu()
            del self._net
            self._net = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
