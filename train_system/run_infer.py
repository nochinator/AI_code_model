from __future__ import annotations

import argparse

import torch

from .model import ModelConfig, TinyCoderLM
from .tokenizer import load_tokenizer
from .utils import choose_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference with trained Codex-style model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer metadata json")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    args = parser.parse_args()

    device = choose_device("auto")
    tokenizer = load_tokenizer(args.tokenizer)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],
        max_seq_len=cfg["data"]["sequence_length"],
        param_budget_millions=cfg["model"].get("param_budget_millions", 175.0),
    )
    model = TinyCoderLM(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, eos_id=tokenizer.eos_id)
    print(tokenizer.decode(out[0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
