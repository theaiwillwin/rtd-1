"""
CWD-PHFT training loop.

Loss:      L_total = L_lm + energy_weight * V.mean()   (energy_weight ~ 0.01)
Optimizer: model.build_optimizer() — identity field projection at lr * 0.01.
Curvature: annealed 0.1 -> 2.0 over 10k steps via model.update_curvature().

Persistent-state protocol (see model/cwd_phft.py):
  detach_persistent_state()  -- after EVERY batch, no exceptions
  reset_persistent_state()   -- ONLY at genuine conversation boundaries
                                (here: once at the start of each epoch)
  decay_override=0.5         -- soft decay when a batch crosses into a
                                new WikiText article (turn-boundary analog)

wandb logging (enable with --wandb): lm_loss, energy, total_loss,
curvature, grad_norm, update_ratio (||delta_theta|| / ||theta|| per
optimizer step), tokens/sec, and every field metric the
PersistentHyperbolicField reports (gate value, delta norms, radii).

Usage:
  python python/training/train.py                          # full config, WikiText-2
  python python/training/train.py --dataset synthetic \
      --steps 20 --dim 64 --num-layers 2 --num-heads 4 \
      --mem-size 16 --vocab-size 1000 --batch-size 2 --seq-len 32   # smoke test
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.cwd_phft import CWDPHFT
from training.data import SequentialLMLoader, load_wikitext2_tokens, synthetic_tokens


def parse_args():
    p = argparse.ArgumentParser(description="Train CWD-PHFT")
    # Established hyperparameters (see model/cwd_phft.py docstring)
    p.add_argument("--vocab-size",    type=int,   default=50257)
    p.add_argument("--dim",           type=int,   default=512)
    p.add_argument("--num-layers",    type=int,   default=6)
    p.add_argument("--num-heads",     type=int,   default=8)
    p.add_argument("--mem-size",      type=int,   default=64)
    p.add_argument("--field-decay",   type=float, default=0.997)
    p.add_argument("--mem-momentum",  type=float, default=0.99)
    p.add_argument("--fractal-iters", type=int,   default=2)
    p.add_argument("--num-modes",     type=int,   default=4)
    p.add_argument("--seq-len",       type=int,   default=256)
    p.add_argument("--batch-size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=0.01)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--energy-weight", type=float, default=0.01)
    # Run control
    p.add_argument("--dataset",    choices=["wikitext", "synthetic"], default="wikitext")
    p.add_argument("--steps",      type=int, default=10000, help="Total optimizer steps")
    p.add_argument("--log-every",  type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000, help="0 disables eval")
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--ckpt-dir",   type=str, default="checkpoints")
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--wandb",      action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="cwd-phft")
    return p.parse_args()


def build_loaders(args):
    if args.dataset == "synthetic":
        tokens, doc_starts = synthetic_tokens(
            n_tokens=max(200 * args.batch_size * args.seq_len, 50_000),
            vocab_size=args.vocab_size,
        )
        train = SequentialLMLoader(tokens, doc_starts, args.batch_size, args.seq_len)
        return train, None
    tokens, doc_starts = load_wikitext2_tokens("train")
    val_tokens, val_doc_starts = load_wikitext2_tokens("validation")
    train = SequentialLMLoader(tokens, doc_starts, args.batch_size, args.seq_len)
    val   = SequentialLMLoader(val_tokens, val_doc_starts, args.batch_size, args.seq_len)
    return train, val


def lm_and_energy_loss(model, ids, decay_override, energy_weight, vocab_size):
    logits, energy = model(ids, decay_override=decay_override, return_energy=True)
    lm_loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1)
    )
    return lm_loss, energy, lm_loss + energy_weight * energy.mean()


@torch.no_grad()
def evaluate(model, loader, args):
    """Validation perplexity. Field/memory state is snapshotted and restored
    so eval never leaks into the training-time persistent state."""
    saved = model.get_field_state_dict()
    model.eval()
    total_nll, total_batches = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= args.eval_batches:
            break
        ids = batch["input_ids"].to(args.device)
        lm_loss, _, _ = lm_and_energy_loss(
            model, ids, 0.5 if batch["new_doc_frac"] > 0 else None,
            args.energy_weight, args.vocab_size,
        )
        model.detach_persistent_state()
        total_nll += lm_loss.item()
        total_batches += 1
    model.load_field_state_dict(saved)
    model.train()
    nll = total_nll / max(total_batches, 1)
    return nll, math.exp(min(nll, 20.0))


def update_ratio(params_before, params_after):
    delta_sq, base_sq = 0.0, 0.0
    for b, a in zip(params_before, params_after):
        delta_sq += (a - b).pow(2).sum().item()
        base_sq  += b.pow(2).sum().item()
    return math.sqrt(delta_sq) / max(math.sqrt(base_sq), 1e-12)


def save_checkpoint(model, optimizer, step, args, tag):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"cwd_phft_{tag}.pt")
    torch.save({
        "step":                 step,
        "model_state_dict":     model.state_dict(),
        "field_state":          model.get_field_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args":                 vars(args),
    }, path)
    return path


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))

    print(f"device={args.device} | dataset={args.dataset} | steps={args.steps}")
    train_loader, val_loader = build_loaders(args)
    print(f"train windows/epoch: {len(train_loader)}"
          + (f" | val windows: {len(val_loader)}" if val_loader else ""))

    model = CWDPHFT(
        vocab_size=args.vocab_size, dim=args.dim, num_layers=args.num_layers,
        num_heads=args.num_heads, mem_size=args.mem_size,
        field_decay=args.field_decay, mem_momentum=args.mem_momentum,
        fractal_iters=args.fractal_iters, num_modes=args.num_modes,
    ).to(args.device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = model.build_optimizer(base_lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    step, t_last, tokens_since = 0, time.time(), 0
    while step < args.steps:
        # Epoch boundary = the one genuine "new conversation" in LM pretraining
        model.reset_persistent_state()
        for batch in train_loader:
            if step >= args.steps:
                break
            ids = batch["input_ids"].to(args.device)
            decay_override = 0.5 if batch["new_doc_frac"] > 0 else None

            lm_loss, energy, loss = lm_and_energy_loss(
                model, ids, decay_override, args.energy_weight, args.vocab_size
            )
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            log_step = step % args.log_every == 0
            if log_step:
                before = [p.detach().clone() for group in optimizer.param_groups
                          for p in group["params"]]
            optimizer.step()

            model.detach_persistent_state()  # after EVERY batch, no exceptions
            model.update_curvature(step)
            step += 1
            tokens_since += ids.numel()

            if log_step:
                after = [p.detach() for group in optimizer.param_groups
                         for p in group["params"]]
                now = time.time()
                metrics = {
                    "lm_loss":      lm_loss.item(),
                    "energy":       energy.mean().item(),
                    "total_loss":   loss.item(),
                    "curvature":    model.current_curvature.item(),
                    "grad_norm":    float(grad_norm),
                    "update_ratio": update_ratio(before, after),
                    "tokens_per_sec": tokens_since / max(now - t_last, 1e-9),
                }
                metrics.update({f"field/{k}": v for k, v in model.field.last_metrics.items()})
                t_last, tokens_since = now, 0
                print(f"step {step:>6} | lm {metrics['lm_loss']:.4f} | "
                      f"E {metrics['energy']:.4f} | c {metrics['curvature']:.3f} | "
                      f"gnorm {metrics['grad_norm']:.3f} | "
                      f"upd {metrics['update_ratio']:.2e} | "
                      f"gate {metrics.get('field/gate_value', float('nan')):.3f} | "
                      f"{metrics['tokens_per_sec']:.0f} tok/s")
                if run:
                    run.log(metrics, step=step)

            if val_loader and args.eval_every and step % args.eval_every == 0:
                val_nll, val_ppl = evaluate(model, val_loader, args)
                print(f"step {step:>6} | val_nll {val_nll:.4f} | val_ppl {val_ppl:.2f}")
                if run:
                    run.log({"val_nll": val_nll, "val_ppl": val_ppl}, step=step)

            if args.ckpt_every and step % args.ckpt_every == 0:
                path = save_checkpoint(model, optimizer, step, args, f"step{step}")
                print(f"step {step:>6} | checkpoint -> {path}")

    path = save_checkpoint(model, optimizer, step, args, "final")
    print(f"done at step {step} | final checkpoint -> {path}")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
