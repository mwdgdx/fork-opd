"""Path B, step 3 (minimal): train the student on constructed fork trajectories.

This first version implements the IMITATION half of the mixed loss — cross-entropy on
the teacher suffix, conditioned on (prompt + student prefix). That fully covers the
WHOLE-REWRITE baseline arm (prefix empty -> student imitates the teacher's full
trajectory given the prompt), and the suffix half of a fork.

TODO (next): the on-policy PREFIX loss (reverse KL between student and the frozen
teacher over the student prefix). It needs teacher top-k logprobs over the prefix,
pre-saved by build_fork_data.py, then combined here via fork_loss.py's reverse-KL
path. Whole-rewrite doesn't need it (no prefix), so the baseline runs today.

Runs on the mcli GPU box.

Example:
    python fork/train.py --data out/fork_whole.jsonl --student Qwen/Qwen3-1.7B \
        --epochs 1 --lr 1e-6 --out checkpoints/whole
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset


class ForkDataset(Dataset):
    """Each item: input_ids = prompt + prefix + suffix; labels supervise the suffix (CE)."""

    def __init__(self, path: str, tokenizer, max_len: int):
        self.items = []
        for line in open(path):
            r = json.loads(line)
            if not r.get("suffix_token_ids"):
                continue  # nofork rows carry no imitation target (need prefix-KL, TODO)
            prompt_ids = tokenizer(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
            prefix_ids = r["prefix_token_ids"]
            suffix_ids = r["suffix_token_ids"]
            input_ids = (prompt_ids + prefix_ids + suffix_ids)[:max_len]
            # -100 on prompt+prefix (context only), supervise suffix positions
            n_ctx = len(prompt_ids) + len(prefix_ids)
            labels = ([-100] * n_ctx + suffix_ids)[:max_len]
            self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lbl in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lbl + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--out", default="checkpoints/run")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.student)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.gradient_checkpointing_enable()   # long fork sequences + full finetune -> avoid OOM
    model.config.use_cache = False
    model.train()

    ds = ForkDataset(args.data, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for ep in range(args.epochs):
        for input_ids, labels, attn in dl:
            input_ids, labels, attn = (t.cuda() for t in (input_ids, labels, attn))
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"epoch {ep} step {step} loss {out.loss.item():.4f}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
