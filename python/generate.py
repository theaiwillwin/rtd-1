"""
Sanity-check generation script for a trained CWD-PHFT checkpoint.

Loads model + field state from a checkpoint, encodes a prompt, and
autoregressively samples a continuation. No KV-cache -- each new token
re-runs the forward pass over the trailing context window, which is
fine for a quick correctness check but not production-speed generation.

Usage:
  python python/generate.py --checkpoint checkpoints/cwd_phft_final.pt \
      --prompt "The history of the internet"
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.cwd_phft import CWDPHFT
from training.data import get_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Generate text from a CWD-PHFT checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="The history of the internet")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--context-window", type=int, default=256,
                   help="Trailing tokens fed to the model each step (matches training seq_len)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k filtering")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, context_window, temperature, top_k, device):
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        window = ids[:, -context_window:]
        logits = model(window)
        next_logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k > 0:
            kth = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[-1]
            next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        model.detach_persistent_state()

    return tokenizer.decode(ids[0].tolist())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = ckpt["args"]

    model = CWDPHFT(
        vocab_size=train_args["vocab_size"], dim=train_args["dim"],
        num_layers=train_args["num_layers"], num_heads=train_args["num_heads"],
        mem_size=train_args["mem_size"], field_decay=train_args["field_decay"],
        mem_momentum=train_args["mem_momentum"], fractal_iters=train_args["fractal_iters"],
        num_modes=train_args["num_modes"],
    ).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.load_field_state_dict(ckpt["field_state"])
    model.eval()

    bad = [k for k, v in ckpt["model_state_dict"].items()
           if torch.is_tensor(v) and not torch.isfinite(v).all()]
    if bad:
        print(f"WARNING: checkpoint has non-finite weights: {bad}")

    print(f"loaded {args.checkpoint} (step {ckpt['step']}, curvature "
          f"{model.current_curvature.item():.3f})")

    tokenizer = get_tokenizer()
    model.reset_persistent_state()  # fresh conversation for this sample
    text = generate(model, tokenizer, args.prompt, args.max_new_tokens,
                    args.context_window, args.temperature, args.top_k, args.device)
    print("---")
    print(text)


if __name__ == "__main__":
    main()
