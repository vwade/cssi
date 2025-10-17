# CML Prototype (Contextless‑State Synthetic Intelligence)

This tiny PyTorch project shows a *stateful* reasoning module ("CMLCell") that
carries forward a compact latent **state** between chunks of tokens, instead of
requiring a huge prompt every step. It also includes a simple **copy/recall**
task that stresses long‑range dependencies while limiting per‑step context.

## Files
- `cml_cell.py` — the CMLCell: gated low‑rank state update + selective KV memory.
- `train_copy_task.py` — synthetic dataset + training loop, evaluates **context efficiency**.
- `README_CML.md` — this file.

## Quick start
```bash
python train_copy_task.py --vocab 128 --seq_min 128 --seq_max 1024 --chunk 32 --d_state 512 --d_model 384 --heads 4 --epochs 2
```

## What to look for
- The model receives a long sequence in **small chunks** (`--chunk`) and must
  reproduce it token‑by‑token. Only a small sliding window is visible; success
  requires retaining information in the **latent state**, not the prompt.
- Metrics:
  - **Context Efficiency**: accuracy vs chunk size / window length.
  - **State Half‑life**: how many steps until a probe on the latent state loses the signal.
  - **Write Sparsity**: fraction of steps that actually write to KV memory.
- Try: increasing `--seq_max`, reducing `--chunk`, freezing the state between batches, etc.
