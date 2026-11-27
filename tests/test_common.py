"""Lightweight checks that do not require a model download or GPU."""
import contextlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import TAGS, encode, generate


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return list(sorted(TAGS.values())).index(token) + 10

    def encode(self, text, **kwargs):
        return [ord(text[0])]

    def pad(self, items, **kwargs):
        return FakeBatch(items=items)

    def batch_decode(self, output, **kwargs):
        return [chr(value) for value in output]


class FakeModel:
    device = "cpu"

    def eval(self):
        pass

    def generate(self, items, **kwargs):
        assert kwargs["use_cache"] is True
        assert "forced_bos_token_id" in kwargs
        return [item["input_ids"][1] for item in items]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.config = {"model_family": "nllb", "max_source_length": 128,
                       "max_target_length": 128, "num_beams": 5, "batch_size": 2}

    def test_source_and_target_language_tags(self):
        row = {"SOURCE_LANG": "fr", "TARGET_LANG": "hav", "SOURCE_TEXT": "bonjour", "TARGET_TEXT": "test"}
        result = encode(row, self.tokenizer, self.config)
        self.assertEqual(result["input_ids"][0], self.tokenizer.convert_tokens_to_ids(TAGS["fr"]))
        self.assertEqual(result["labels"][0], self.tokenizer.convert_tokens_to_ids(TAGS["hav"]))

    def test_length_sorting_restores_row_order(self):
        rows = [
            {"SOURCE_LANG": "fr", "TARGET_LANG": "hav", "SOURCE_TEXT": "aaaa"},
            {"SOURCE_LANG": "fr", "TARGET_LANG": "shr", "SOURCE_TEXT": "b"},
            {"SOURCE_LANG": "fr", "TARGET_LANG": "hav", "SOURCE_TEXT": "c"},
        ]
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = generate(rows, FakeModel(), self.tokenizer, self.config)
        self.assertEqual(result, ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
