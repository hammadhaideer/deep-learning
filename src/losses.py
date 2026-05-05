import torch


def reconstruction_loss(tokens: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    return (tokens - recon).pow(2).mean()
