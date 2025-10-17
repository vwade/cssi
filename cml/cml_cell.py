
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class CMLCell(nn.Module):
    """
    A compact 'Contextless‑State' block.
    - Maintains a latent state s_t in R^{B x D_state}
    - Reads a small token chunk x_t in R^{B x T x D_model}
    - Emits next‑token logits and an updated state
    Ingredients:
      * Gated low‑rank update (GLRU) for s_{t+1}
      * Selective key‑value memory with learned write gate
      * Lightweight attention from x_t to (state + memory)
    """
    def __init__(self, vocab_size: int, d_model: int = 384, d_state: int = 512, heads: int = 4, mem_slots: int = 32):
        super().__init__()
        self.vocab = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.heads = heads
        self.mem_slots = mem_slots

        # Token embeddings + projection
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.in_proj = nn.Linear(d_model, d_model)

        # Project state into query/key/value spaces
        self.state_to_k = nn.Linear(d_state, d_model)
        self.state_to_v = nn.Linear(d_state, d_model)

        # Memory: keys/values + learnable initial slots per head
        self.mem_k = nn.Parameter(torch.randn(1, heads, mem_slots, d_model // heads) * 0.02)
        self.mem_v = nn.Parameter(torch.randn(1, heads, mem_slots, d_model // heads) * 0.02)

        # Attention for tokens -> (state, memory)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Gated Low‑Rank Update (GLRU) for the latent state
        rank = max(16, d_state // 16)
        self.U = nn.Linear(d_state, rank, bias=False)
        self.V = nn.Linear(rank, d_state, bias=False)
        self.gate_upd = nn.Linear(d_model + d_state, d_state)

        # Write gate for memory (sparse updates)
        self.write_gate = nn.Linear(d_model + d_state, heads)

        # Decoder for next‑token logits
        self.ln = nn.LayerNorm(d_model)
        self.dec = nn.Linear(d_model, vocab_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.heads
        return x.view(B, T, H, D // H).transpose(1, 2)  # B,H,T,dh

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * dh)

    def forward(self, x_tok: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_tok: (B, T) integer tokens
        state: (B, D_state) latent; if None -> zeros
        Returns: (logits: B,T,V), (state_next: B,D_state)
        """
        B, T = x_tok.shape
        if state is None:
            state = torch.zeros(B, self.d_state, device=x_tok.device)

        x = self.tok_emb(x_tok)
        x = self.in_proj(x)

        # Build attention keys/values from state + memory
        s_k = self.state_to_k(state).unsqueeze(1)              # B,1,D
        s_v = self.state_to_v(state).unsqueeze(1)              # B,1,D
        s_k = s_k.expand(B, 1, -1)
        s_v = s_v.expand(B, 1, -1)

        # Project tokens
        q = self._split_heads(self.q_proj(x))                  # B,H,T,dh
        k_tokens = self._split_heads(self.k_proj(x))           # B,H,T,dh
        v_tokens = self._split_heads(self.v_proj(x))           # B,H,T,dh

        # Keys/Values: [state slot] + [learned memory slots] + [local tokens]
        k_state = self._split_heads(s_k)                       # B,H,1,dh
        v_state = self._split_heads(s_v)                       # B,H,1,dh
        k_mem = self.mem_k.expand(B, -1, -1, -1)               # B,H,M,dh
        v_mem = self.mem_v.expand(B, -1, -1, -1)               # B,H,M,dh

        K = torch.cat([k_state, k_mem, k_tokens], dim=2)       # B,H,1+M+T,dh
        V = torch.cat([v_state, v_mem, v_tokens], dim=2)       # B,H,1+M+T,dh

        # Scaled dot‑product attention
        att = (q * (1.0 / (K.shape[-1] ** 0.5))) @ K.transpose(-2, -1)   # B,H,T,1+M+T
        att = att.softmax(dim=-1)
        ctx = att @ V                                                    # B,H,T,dh
        y = self._merge_heads(ctx)                                       # B,T,D
        y = self.out_proj(self.ln(y))                                    # B,T,D

        # Next‑token logits
        logits = self.dec(y)                                             # B,T,V

        # Gated low‑rank state update driven by last token context
        last_ctx = y[:, -1, :]                                           # B,D
        low_rank = self.V(torch.tanh(self.U(state)))                     # B,D_state
        gate = torch.sigmoid(self.gate_upd(torch.cat([last_ctx, state], dim=-1)))
        state_next = (1.0 - gate) * state + gate * (state + low_rank)

        # Sparse write into memory slots (optional simple heuristic)
        # Compute a per‑head write scalar; move a little of state into mem_v
        write_alpha = torch.sigmoid(self.write_gate(torch.cat([last_ctx, state], dim=-1)))  # B,H
        write_alpha = write_alpha.unsqueeze(-1).unsqueeze(-1)                                # B,H,1,1
        with torch.no_grad():
            self.mem_v.copy_(0.98 * self.mem_v + 0.02 * (write_alpha.mean(0, keepdim=True) * self.mem_v))

        return logits, state_next
