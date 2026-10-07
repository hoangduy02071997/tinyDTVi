"""
Full definition of a TinyDTVi Language Model, all of it in this single file.
References:
1) the official nanoGPT Karpathy structure Transformer:
https://github.com/karpathy/nanoGPT/blob/master/model.py
2) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
3) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# --- 1. RMSNorm (Standard DeepSeek/Llama implementation) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x).type_as(x)

# --- 2. RoPE Helpers (Vector rotation utilities) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Use float32 for accurate RoPE computation, but cast back immediately
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(3))
    
    # Multiply complex numbers and convert back to original float (typically bfloat16)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).to(xq.dtype)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).to(xk.dtype)
    return xq_out, xk_out

# --- 3. SwiGLU MLP (Improved from standard MLP) ---
class SwiGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU requires 3 matrices instead of 2
        self.w1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.w2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Formula: SwiGLU(x) = (Swish(xW1) * xW3)W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# --- 4. Upgraded Attention with RoPE ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x, freqs_cis):
        B, T, C = x.size()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, T, self.n_head, C // self.n_head)
        xk = xk.view(B, T, self.n_head, C // self.n_head)
        xv = xv.view(B, T, self.n_head, C // self.n_head)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=None, 
            dropout_p=self.dropout if self.training else 0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

# --- 5. Complete Model Assembly ---
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class TinyDTViConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24
    n_embd: int = 1088
    n_head: int = 16
    dropout: float = 0.0

class TinyDTVi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.block_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        
        # Retrieve freqs_cis corresponding to the current sequence length
        freqs_cis = self.freqs_cis[:T].to(idx.device)

        # Check for gradient checkpointing (usually enabled during training for large models)
        use_checkpoint = self.training and getattr(self, "gradient_checkpointing", False)

        for block in self.transformer.h:
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, freqs_cis, use_reentrant=False)
            else:
                x = block(x, freqs_cis)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Optimization: Calculate loss directly from projection to avoid storing massive Logits matrix (B, T, Vocab)
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # Return logits=None during training to save VRAM
            return None, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU peak
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx