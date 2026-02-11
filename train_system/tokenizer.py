from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class TokenizerLike(Protocol):
    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...

    def save(self, path: str) -> None: ...

    @property
    def vocab_size(self) -> int: ...


@dataclass
class SimpleTokenizer:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        pieces = text.split()
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.token_to_id.get(piece, self.unk_id) for piece in pieces)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = []
        for idx in ids:
            tok = self.id_to_token[idx]
            if tok in SPECIAL_TOKENS:
                continue
            toks.append(tok)
        return " ".join(toks)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "simple", "id_to_token": self.id_to_token}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        id_to_token = payload["id_to_token"]
        token_to_id = {t: i for i, t in enumerate(id_to_token)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError("sentencepiece is required for sentencepiece tokenizer") from exc
        self._sp = spm.SentencePieceProcessor(model_file=model_file)
        self.model_file = model_file

    @property
    def pad_id(self) -> int:
        return int(self._sp.pad_id())

    @property
    def bos_id(self) -> int:
        return int(self._sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self._sp.eos_id())

    @property
    def vocab_size(self) -> int:
        return int(self._sp.vocab_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = list(self._sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        filtered = [i for i in ids if i not in {self.pad_id, self.bos_id, self.eos_id}]
        return self._sp.decode(filtered)

    def save(self, path: str) -> None:
        meta = {"format": "sentencepiece", "model_file": self.model_file}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def train_simple_tokenizer(texts: Iterable[str], vocab_size: int = 16000, min_freq: int = 2) -> SimpleTokenizer:
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = list(SPECIAL_TOKENS)
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab.append(token)
        if len(vocab) >= vocab_size:
            break

    token_to_id = {t: i for i, t in enumerate(vocab)}
    return SimpleTokenizer(token_to_id=token_to_id, id_to_token=vocab)


def train_sentencepiece_tokenizer(
    texts: Iterable[str],
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "bpe",
) -> SentencePieceTokenizer:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError("Install sentencepiece to use tokenizer.type=sentencepiece") from exc

    prefix_path = Path(model_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = prefix_path.parent / f"{prefix_path.name}_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(prefix_path),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    model_file = str(prefix_path) + ".model"
    return SentencePieceTokenizer(model_file=model_file)


def build_tokenizer(texts: Iterable[str], cfg: Dict[str, object]) -> TokenizerLike:
    tokenizer_type = str(cfg.get("type", "sentencepiece")).lower()
    if tokenizer_type == "simple":
        return train_simple_tokenizer(
            texts,
            vocab_size=int(cfg.get("vocab_size", 16000)),
            min_freq=int(cfg.get("min_freq", 2)),
        )
    if tokenizer_type == "sentencepiece":
        return train_sentencepiece_tokenizer(
            texts,
            model_prefix=str(cfg.get("model_prefix", "outputs/tokenizer")),
            vocab_size=int(cfg.get("vocab_size", 32000)),
            model_type=str(cfg.get("model_type", "bpe")),
        )
    raise ValueError(f"Unsupported tokenizer.type: {tokenizer_type}")


def load_tokenizer(meta_path: str) -> TokenizerLike:
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    fmt = payload.get("format")
    if fmt == "simple":
        id_to_token = payload["id_to_token"]
        return SimpleTokenizer(token_to_id={t: i for i, t in enumerate(id_to_token)}, id_to_token=id_to_token)
    if fmt == "sentencepiece":
        return SentencePieceTokenizer(model_file=payload["model_file"])
    raise ValueError(f"Unknown tokenizer format in {meta_path}: {fmt}")
