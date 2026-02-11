import tempfile
import unittest


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(torch_available(), "torch is required for training system tests")
class TestTrainingSystem(unittest.TestCase):
    def test_data_generation_and_simple_tokenizer(self):
        from train_system.data import generate_synthetic_jsonl, load_jsonl_text
        from train_system.tokenizer import train_simple_tokenizer

        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/train.jsonl"
            generate_synthetic_jsonl(path, n_examples=20, seed=123)
            texts = load_jsonl_text(path)
            self.assertEqual(len(texts), 20)

            tok = train_simple_tokenizer(texts, vocab_size=256, min_freq=1)
            ids = tok.encode(texts[0])
            self.assertGreater(len(ids), 5)
            decoded = tok.decode(ids)
            self.assertIsInstance(decoded, str)

    def test_model_forward(self):
        from train_system.data import TokenDataset, collate_batch
        from train_system.model import ModelConfig, TinyCoderLM, count_parameters
        from train_system.tokenizer import train_simple_tokenizer

        texts = ["<|system|> hello <|user|> build api <|assistant|> def solve(): pass"]
        tok = train_simple_tokenizer(texts, vocab_size=128, min_freq=1)
        seqs = [tok.encode(t) for t in texts]
        ds = TokenDataset(seqs, sequence_length=32)
        batch = collate_batch([ds[0]], pad_id=tok.pad_id)

        cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=64)
        model = TinyCoderLM(cfg)
        logits = model(batch["input_ids"])
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[-1], tok.vocab_size)
        self.assertGreater(count_parameters(model), 1000)

    def test_generate(self):
        import torch

        from train_system.model import ModelConfig, TinyCoderLM
        from train_system.tokenizer import train_simple_tokenizer

        texts = ["<|system|> a <|user|> b <|assistant|> c"]
        tok = train_simple_tokenizer(texts, vocab_size=64, min_freq=1)
        cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_seq_len=64)
        model = TinyCoderLM(cfg)
        x = torch.tensor([tok.encode(texts[0])], dtype=torch.long)
        out = model.generate(x, max_new_tokens=4, eos_id=tok.eos_id)
        self.assertGreaterEqual(out.shape[1], x.shape[1])


if __name__ == "__main__":
    unittest.main()
