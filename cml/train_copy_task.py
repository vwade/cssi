
import argparse, math, random, os
import torch
import torch.nn as nn
import torch.optim as optim
from cml_cell import CMLCell

# ---- Synthetic copy/recall dataset ----
def make_seq(vocab: int, n: int):
    # tokens 2..(vocab-1) for data, 1 as BOS, 0 as PAD (not used)
    return [1] + [random.randint(2, vocab - 1) for _ in range(n)]

def batchify(seqs, pad_id=0):
    mx = max(len(s) for s in seqs)
    x = torch.full((len(seqs), mx), pad_id, dtype=torch.long)
    for i,s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return x

def iter_data(vocab, seq_min, seq_max, batch, steps):
    for _ in range(steps):
        lengths = [random.randint(seq_min, seq_max) for _ in range(batch)]
        seqs = [make_seq(vocab, n) for n in lengths]
        x = batchify(seqs)
        # Predict next token (teacher forcing); target is x shifted left
        y = x.clone()
        y[:, :-1] = x[:, 1:]
        return_x, return_y = x, y
        yield return_x, return_y

# ---- Training ----
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMLCell(vocab_size=args.vocab, d_model=args.d_model, d_state=args.d_state, heads=args.heads).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in iter_data(args.vocab, args.seq_min, args.seq_max, args.batch, args.steps_per_epoch):
            xb, yb = xb.to(device), yb.to(device)
            # Stream in small chunks to force reliance on latent state
            B, T = xb.shape
            state = None
            total_loss = 0.0
            denom = 0
            for t0 in range(0, T, args.chunk):
                x_chunk = xb[:, t0:t0+args.chunk]
                y_chunk = yb[:, t0:t0+args.chunk]
                logits, state = model(x_chunk, state)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_chunk.reshape(-1))
                total_loss += loss
                denom += 1
            opt.zero_grad(set_to_none=True)
            (total_loss / max(1, denom)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    ppl = torch.exp(total_loss / max(1, denom))
                print(f"ep {epoch} step {global_step} | loss {total_loss.item():.3f} | ppl {ppl.item():.2f} | chunk {args.chunk} | seq~{(args.seq_min+args.seq_max)//2}")
            global_step += 1

        # quick eval: vary chunk size to probe context efficiency
        model.eval()
        for probe_chunk in [args.chunk, max(8, args.chunk//2), args.chunk*2]:
            with torch.no_grad():
                xb, yb = next(iter_data(args.vocab, args.seq_max, args.seq_max, args.batch, 1))
                xb, yb = xb.to(device), yb.to(device)
                state = None
                losses = []
                B, T = xb.shape
                for t0 in range(0, T, probe_chunk):
                    x_chunk = xb[:, t0:t0+probe_chunk]
                    y_chunk = yb[:, t0:t0+probe_chunk]
                    logits, state = model(x_chunk, state)
                    loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_chunk.reshape(-1))
                    losses.append(loss.item())
                print(f"[eval] chunk={probe_chunk} avg_loss={sum(losses)/len(losses):.3f}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--seq_min", type=int, default=128)
    ap.add_argument("--seq_max", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--d_state", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()
    run(args)
