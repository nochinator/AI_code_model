from __future__ import annotations

import argparse
import json

from .trainer import Trainer
from .utils import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Codex-style assistant language model")
    parser.add_argument("--config", default="configs/train_base.yaml", help="Path to training config")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    trainer = Trainer(cfg)
    metrics = trainer.train()
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
