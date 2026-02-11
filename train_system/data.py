from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SYSTEM_TEMPLATES = [
    "You are a thoughtful assistant who explains your reasoning and tradeoffs.",
    "You are an expert coding assistant. Provide robust, testable answers.",
    "You are a multilingual assistant for software, math, and general knowledge.",
]

USER_TEMPLATES = [
    "Implement {feature} in {language} and include a brief test strategy.",
    "Explain how {topic} works, then provide a {language} example.",
    "Debug this failing approach for {feature} and show a corrected version.",
    "Compare two approaches for {topic} and recommend one with reasons.",
]

ASSISTANT_TEMPLATES = [
    "Approach:\n- Clarify requirements\n- Build a safe baseline\n- Add tests\n\nImplementation:\n```{language}\n{code}\n```\n",
    "Good choice. Here's a clear {language} solution with notes:\n```{language}\n{code}\n```\nKey tradeoffs: simplicity vs. scalability.",
]

TOPICS = [
    "retrieval-augmented generation",
    "API design",
    "distributed queues",
    "database indexing",
    "security hardening",
    "model quantization",
    "evaluation metrics",
]

FEATURES = [
    "authentication middleware",
    "rate limiting",
    "file upload validation",
    "streaming inference",
    "retry logic with exponential backoff",
    "structured logging",
    "role-based access control",
    "caching layer",
]

LANGUAGES = ["python", "javascript", "rust", "sql", "go"]

CODE_SNIPPETS = {
    "python": "def solve(x):\n    return {'ok': True, 'value': x}\n",
    "javascript": "function solve(x){ return { ok: true, value: x }; }\n",
    "rust": "fn solve(x: &str) -> String { format!(\"ok:{}\", x) }\n",
    "sql": "SELECT id, status FROM jobs WHERE status='queued' ORDER BY id DESC LIMIT 100;\n",
    "go": "func Solve(x string) map[string]any { return map[string]any{" + '"ok": true, "value": x}' + " }\n",
}


def _sample_synthetic_example(rng: random.Random) -> str:
    topic = rng.choice(TOPICS)
    feature = rng.choice(FEATURES)
    language = rng.choice(LANGUAGES)

    system = rng.choice(SYSTEM_TEMPLATES)
    user = rng.choice(USER_TEMPLATES).format(feature=feature, language=language, topic=topic)
    assistant = rng.choice(ASSISTANT_TEMPLATES).format(language=language, code=CODE_SNIPPETS[language])
    return (
        "<|system|>\n"
        f"{system}\n"
        "<|user|>\n"
        f"{user}\n"
        "<|assistant|>\n"
        f"{assistant}"
    )


def generate_synthetic_jsonl(path: str, n_examples: int, seed: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n_examples):
            f.write(json.dumps({"text": _sample_synthetic_example(rng)}, ensure_ascii=False) + "\n")


def load_jsonl_text(path: str, text_key: str = "text", max_rows: int | None = None) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            if text_key in payload and isinstance(payload[text_key], str) and payload[text_key].strip():
                rows.append(payload[text_key])
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def maybe_load_hf_dataset(source_cfg: Dict[str, object], seed: int) -> List[str]:
    """Load text rows from HuggingFace datasets when available.

    Requires `datasets` package and network access. Falls back to [] if unavailable.
    """

    try:
        from datasets import load_dataset
    except Exception:
        return []

    dataset_name = str(source_cfg["dataset"])
    split = str(source_cfg.get("split", "train"))
    text_field = str(source_cfg.get("text_field", "text"))
    subset = source_cfg.get("subset")
    max_samples = int(source_cfg.get("max_samples", 0))

    try:
        if subset:
            ds = load_dataset(dataset_name, str(subset), split=split)
        else:
            ds = load_dataset(dataset_name, split=split)
    except Exception:
        return []

    if max_samples > 0 and max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    texts = []
    for row in ds:
        val = row.get(text_field)
        if isinstance(val, str) and val.strip():
            texts.append(val)
    return texts


def build_corpus(data_cfg: Dict[str, object], seed: int, split: str) -> List[str]:
    """Build a training/eval corpus from configured external + synthetic sources."""

    rng = random.Random(seed)
    sources: List[Dict[str, object]] = list(data_cfg.get(f"{split}_sources", []))
    corpus: List[str] = []

    for source in sources:
        source_type = str(source.get("type", "jsonl"))
        if source_type == "jsonl":
            path = str(source["path"])
            if Path(path).exists():
                corpus.extend(load_jsonl_text(path, text_key=str(source.get("text_key", "text")), max_rows=source.get("max_rows")))
        elif source_type == "huggingface":
            corpus.extend(maybe_load_hf_dataset(source, seed=seed))

    synthetic_target = int(data_cfg.get(f"{split}_synthetic_examples", 0))
    for _ in range(synthetic_target):
        corpus.append(_sample_synthetic_example(rng))

    rng.shuffle(corpus)
    return corpus


class TokenDataset(Dataset):
    def __init__(self, token_sequences: Sequence[List[int]], sequence_length: int) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        for seq in token_sequences:
            if len(seq) < 2:
                continue
            for i in range(0, len(seq) - 1, sequence_length):
                chunk = seq[i : i + sequence_length + 1]
                if len(chunk) >= 2:
                    self.samples.append((chunk[:-1], chunk[1:]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_batch(batch, pad_id: int):
    max_len = max(len(x[0]) for x in batch)
    x_out, y_out, mask = [], [], []
    for x, y in batch:
        x_pad = x + [pad_id] * (max_len - len(x))
        y_pad = y + [pad_id] * (max_len - len(y))
        m = [1] * len(x) + [0] * (max_len - len(x))
        x_out.append(x_pad)
        y_out.append(y_pad)
        mask.append(m)
    return {
        "input_ids": torch.tensor(x_out, dtype=torch.long),
        "labels": torch.tensor(y_out, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.float32),
    }


def build_dataloaders(train_ds: TokenDataset, val_ds: TokenDataset, batch_size: int, pad_id: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda b: collate_batch(b, pad_id))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=lambda b: collate_batch(b, pad_id))
    return train_loader, val_loader
