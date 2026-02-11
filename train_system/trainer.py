from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import TokenDataset, build_corpus, build_dataloaders
from .model import ModelConfig, TinyCoderLM, count_parameters
from .tokenizer import build_tokenizer, load_tokenizer
from .utils import TrainState, autocast_context, choose_device, ensure_dir, save_json, set_seed


class Trainer:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        set_seed(cfg["seed"])
        ensure_dir(cfg["output_dir"])
        self.device = choose_device(cfg["device"])

        train_texts = build_corpus(cfg["data"], seed=cfg["seed"], split="train")
        val_texts = build_corpus(cfg["data"], seed=cfg["seed"] + 1, split="val")
        if not train_texts:
            raise ValueError("No training data found. Add train_sources entries or train_synthetic_examples.")
        if not val_texts:
            raise ValueError("No validation data found. Add val_sources entries or val_synthetic_examples.")

        tokenizer_meta_path = cfg["tokenizer"]["meta_path"]
        if Path(tokenizer_meta_path).exists() and not cfg["tokenizer"].get("retrain", True):
            self.tokenizer = load_tokenizer(tokenizer_meta_path)
        else:
            self.tokenizer = build_tokenizer(train_texts, cfg["tokenizer"])
            self.tokenizer.save(tokenizer_meta_path)

        train_tokens = [self.tokenizer.encode(t) for t in train_texts]
        val_tokens = [self.tokenizer.encode(t) for t in val_texts]

        seq_len = cfg["data"]["sequence_length"]
        self.train_ds = TokenDataset(train_tokens, sequence_length=seq_len)
        self.val_ds = TokenDataset(val_tokens, sequence_length=seq_len)

        self.train_loader, self.val_loader = build_dataloaders(
            self.train_ds,
            self.val_ds,
            batch_size=cfg["training"]["micro_batch_size"],
            pad_id=self.tokenizer.pad_id,
        )

        model_cfg = ModelConfig(
            vocab_size=self.tokenizer.vocab_size,
            d_model=cfg["model"]["d_model"],
            n_heads=cfg["model"]["n_heads"],
            n_layers=cfg["model"]["n_layers"],
            d_ff=cfg["model"]["d_ff"],
            dropout=cfg["model"]["dropout"],
            max_seq_len=seq_len,
            param_budget_millions=cfg["model"].get("param_budget_millions", 175.0),
        )
        self.model = TinyCoderLM(model_cfg).to(self.device)

        param_count = count_parameters(self.model)
        param_m = param_count / 1_000_000
        budget_m = float(model_cfg.param_budget_millions)
        if param_m > budget_m * 1.05:
            raise ValueError(f"Model has {param_m:.2f}M params, above budget {budget_m:.2f}M")

        if cfg["training"].get("compile_model", False) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"]["lr"],
            betas=(0.9, 0.95),
            weight_decay=cfg["training"]["weight_decay"],
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and cfg["training"].get("mixed_precision", True)))
        self.state = TrainState()
        self.accum_steps = max(1, cfg["training"]["batch_size"] // cfg["training"]["micro_batch_size"])
        self.param_count = param_count

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        vocab = logits.size(-1)
        loss = F.cross_entropy(logits.view(-1, vocab), labels.view(-1), reduction="none")
        loss = loss.view(labels.size())
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def _lr(self, step: int) -> float:
        warmup = self.cfg["training"]["warmup_steps"]
        max_steps = self.cfg["training"]["max_steps"]
        base = self.cfg["training"]["lr"]
        min_lr_scale = self.cfg["training"].get("min_lr_scale", 0.1)
        if step < warmup:
            return base * (step + 1) / warmup
        progress = (step - warmup) / max(1, max_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * (min_lr_scale + (1 - min_lr_scale) * cosine)

    @torch.no_grad()
    def evaluate(self, max_batches: int = 200) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            x = batch["input_ids"].to(self.device)
            y = batch["labels"].to(self.device)
            m = batch["attention_mask"].to(self.device)
            logits = self.model(x)
            losses.append(self._loss(logits, y, m).item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    def save_checkpoint(self, name: str) -> None:
        ckpt_dir = Path(self.cfg["output_dir"]) / "checkpoints"
        ensure_dir(ckpt_dir)
        path = ckpt_dir / f"{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.state.step,
                "best_val_loss": self.state.best_val_loss,
                "param_count": self.param_count,
                "config": self.cfg,
            },
            path,
        )

    def train(self) -> Dict[str, Any]:
        max_steps = self.cfg["training"]["max_steps"]
        log_every = self.cfg["training"]["log_every"]
        eval_every = self.cfg["training"]["eval_every"]
        save_every = self.cfg["training"]["save_every"]
        grad_clip = self.cfg["training"]["grad_clip"]

        iterator = itertools.cycle(self.train_loader)
        start = time.time()
        running = 0.0

        pbar = tqdm(total=max_steps, desc="training", dynamic_ncols=True)
        while self.state.step < max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(self.accum_steps):
                batch = next(iterator)
                x = batch["input_ids"].to(self.device)
                y = batch["labels"].to(self.device)
                m = batch["attention_mask"].to(self.device)

                with autocast_context(self.device, self.cfg["training"].get("mixed_precision", True)):
                    logits = self.model(x)
                    loss = self._loss(logits, y, m) / self.accum_steps

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            lr = self._lr(self.state.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.state.step += 1
            running += step_loss
            pbar.update(1)

            if self.state.step % log_every == 0:
                avg = running / log_every
                running = 0.0
                pbar.set_postfix(step=self.state.step, train_loss=f"{avg:.4f}", lr=f"{lr:.2e}")

            if self.state.step % eval_every == 0:
                val_loss = self.evaluate(max_batches=self.cfg["training"].get("eval_batches", 200))
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self.save_checkpoint("best")
                pbar.write(f"step={self.state.step} val_loss={val_loss:.4f} best={self.state.best_val_loss:.4f}")

            if self.state.step % save_every == 0:
                self.save_checkpoint(f"step_{self.state.step}")

        pbar.close()
        self.save_checkpoint("final")
        elapsed = time.time() - start

        metrics = {
            "steps": self.state.step,
            "best_val_loss": self.state.best_val_loss,
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600,
            "parameters": self.param_count,
            "parameters_millions": self.param_count / 1_000_000,
            "train_samples": len(self.train_ds),
            "val_samples": len(self.val_ds),
        }
        save_json(Path(self.cfg["output_dir"]) / "metrics.json", metrics)
        return metrics
