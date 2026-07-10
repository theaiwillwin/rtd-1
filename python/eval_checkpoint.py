"""
Compute validation perplexity for a saved checkpoint after the fact --
useful when the original training console output wasn't saved to a file.

Usage:
  python python/eval_checkpoint.py --checkpoint checkpoints/cwd_phft_final.pt
  python python/eval_checkpoint.py --checkpoint checkpoints/cwd_phft_step1000.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.cwd_phft import CWDPHFT
from training.data import SequentialLMLoader, load_wikitext2_tokens
from training.train import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a CWD-PHFT checkpoint on WikiText-2 validation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--eval-batches", type=int, default=10_000, help="High default covers the full val set")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    cli_args = parse_args()
    ckpt = torch.load(cli_args.checkpoint, map_location=cli_args.device, weights_only=False)
    train_args = ckpt["args"]

    model = CWDPHFT(
        vocab_size=train_args["vocab_size"], dim=train_args["dim"],
        num_layers=train_args["num_layers"], num_heads=train_args["num_heads"],
        mem_size=train_args["mem_size"], field_decay=train_args["field_decay"],
        mem_momentum=train_args["mem_momentum"], fractal_iters=train_args["fractal_iters"],
        num_modes=train_args["num_modes"],
    ).to(cli_args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.load_field_state_dict(ckpt["field_state"])

    val_tokens, val_doc_starts = load_wikitext2_tokens("validation")
    val_loader = SequentialLMLoader(val_tokens, val_doc_starts,
                                    train_args["batch_size"], train_args["seq_len"])

    eval_args = argparse.Namespace(
        eval_batches=cli_args.eval_batches, device=cli_args.device,
        energy_weight=train_args["energy_weight"], vocab_size=train_args["vocab_size"],
        energy_reg=train_args["energy_reg"],
    )
    val_nll, val_ppl = evaluate(model, val_loader, eval_args)
    print(f"{cli_args.checkpoint} | step {ckpt['step']} | "
          f"val_nll {val_nll:.4f} | val_ppl {val_ppl:.2f} | "
          f"windows evaluated: {min(cli_args.eval_batches, len(val_loader))}/{len(val_loader)}")


if __name__ == "__main__":
    main()
