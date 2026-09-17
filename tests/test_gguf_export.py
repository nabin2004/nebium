import tempfile
from pathlib import Path
import pytest
import torch
import gguf

from src.models.transformer.nebium import Nebium
from src.export.gguf_export import export_to_gguf


class DummyTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4, "e2e4": 5, "e7e5": 6}
        self.vocab_size = len(self.vocab)
        self.pad_id = 0
        self.bos_id = 2
        self.eos_id = 3
        self.sep_id = 4

    def get_vocab(self):
        return self.vocab

    def encode(self, text):
        return [self.vocab.get(w, 1) for w in text.split()]

    def decode(self, ids):
        id_to_word = {v: k for k, v in self.vocab.items()}
        return " ".join(id_to_word.get(i, "[UNK]") for i in ids)


def test_gguf_export_fp16():
    model = Nebium(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        max_seq_len=64,
        activation="swiglu",
        norm="rmsnorm",
    )
    tokenizer = DummyTokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "nebium_test.gguf"
        export_to_gguf(
            model=model,
            tokenizer=tokenizer,
            output_path=out_path,
            precision="fp16",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

        # Read back header and verify magic bytes
        with open(out_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GGUF"

        # Verify using GGUFReader
        reader = gguf.GGUFReader(str(out_path))
        assert reader.get_field("general.architecture") is not None
        assert reader.get_field("nebium.context_length") is not None
        assert len(reader.tensors) > 0
        del reader
        import gc
        gc.collect()


def test_gguf_export_fp32():
    model = Nebium(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dropout=0.0,
        max_seq_len=32,
    )
    tokenizer = DummyTokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "nebium_fp32.gguf"
        export_to_gguf(
            model=model,
            tokenizer=tokenizer,
            output_path=out_path,
            precision="fp32",
        )

        assert out_path.exists()
        with open(out_path, "rb") as f:
            assert f.read(4) == b"GGUF"
