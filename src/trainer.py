import math
import os
import time
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter

from .losses import reconstruction_loss


class Trainer:
    def __init__(self, model, train_loader, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = device

        self.epochs = cfg["train"]["epochs"]
        self.warmup_epochs = cfg["train"]["warmup_epochs"]
        self.base_lr = cfg["train"]["lr"]
        self.amp = cfg["train"]["amp"] and device.startswith("cuda")

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.base_lr,
            weight_decay=cfg["train"]["weight_decay"],
        )
        self.scaler = GradScaler(enabled=self.amp)

        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(cfg["paths"]["log_dir"])

    def _lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
        return 0.5 * self.base_lr * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def fit(self):
        log_every = self.cfg["train"]["log_every"]
        save_every = self.cfg["train"]["save_every"]
        global_step = 0

        for epoch in range(self.epochs):
            self.model.train()
            self._set_lr(self._lr(epoch))
            epoch_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for batch in self.train_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.amp):
                    tokens, recon = self.model(images)
                    loss = reconstruction_loss(tokens, recon)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % log_every == 0:
                    self.writer.add_scalar("train/loss", loss.item(), global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], global_step)

            avg_loss = epoch_loss / max(n_batches, 1)
            self.writer.add_scalar("train/epoch_loss", avg_loss, epoch)
            print(
                f"epoch {epoch+1:4d}/{self.epochs} | loss {avg_loss:.6f} "
                f"| lr {self.optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s"
            )

            if (epoch + 1) % save_every == 0 or (epoch + 1) == self.epochs:
                self.save(self.ckpt_dir / f"uniad_epoch{epoch+1}.pth", epoch)

        self.writer.close()

    def save(self, path: Path, epoch: int):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"saved {path}")
