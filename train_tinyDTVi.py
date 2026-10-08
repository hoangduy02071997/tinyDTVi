import os
import time
import math
import gc
import torch
import csv
import numpy as np
import glob
from contextlib import nullcontext
from tinyDTVi_model import TinyDTViConfig, TinyDTVi
from transformers import AutoTokenizer


# --- SYSTEM CONFIGURATION ---
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:64'

# -----------------------------------------------------------------------------
out_dir = 'tinyDTVi-500M'
eval_interval = 250
log_interval = 1
eval_iters = 200
always_save_checkpoint = True
save_interval = 100 # Increased to 100 to avoid I/O bottlenecks

# Auto Resume
ckpt_files = glob.glob(os.path.join(out_dir, 'ckpt_*.pt'))
ckpt_iters = []
for f in ckpt_files:
    try:
        it = int(os.path.basename(f).replace('ckpt_', '').replace('.pt', ''))
        ckpt_iters.append((f, it))
    except ValueError:
        continue

ckpt_iters.sort(key=lambda x: x[1], reverse=True)
init_from = 'resume' if len(ckpt_iters) > 0 else 'scratch'

# Model Config from model.py
from tinyDTVi_model import TinyDTViConfig
config = TinyDTViConfig() # Getting n_layer=24, n_embd=1088 from here
n_layer = config.n_layer
n_head = config.n_head
n_embd = config.n_embd
block_size = config.block_size
vocab_size = config.vocab_size
dropout = 0.1

# Data & Training
batch_size = 1 
gradient_accumulation_steps = 80 
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

# LR Decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

device = 'cuda'
dtype = 'bfloat16' 
compile = False

# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
log_file = os.path.join(out_dir, 'loss_log.csv')

# Initialize log file
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iter', 'train_loss', 'val_loss', 'lr'])

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

# Data loader (memmap)
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    # Avoid using pin_memory to prevent Segfaults on Ubuntu 24.04
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

# Model Init
iter_num = 0
best_val_loss = 1e9

if init_from == 'scratch':
    print("Init TinyDTVi-500M model from scratch...")
    model = TinyDTVi(config)
else:
    loaded = False
    for ckpt_path, it in ckpt_iters:
        print(f"Attempting to resume from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model = TinyDTVi(TinyDTViConfig(**checkpoint['model_args']))
            state_dict = checkpoint['model']
            # Fix prefix if the model was compiled previously
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            
            # --- NEW: Restore RNG state ---
            if 'rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['rng_state'])
            if 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            

            
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}")
            # Do NOT delete the file if it's a memory error
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                print("Memory error detected. Halting resume process to protect checkpoints.")
                break
            
            import os
            try:
                os.remove(ckpt_path)
                print(f"Deleted corrupted file {ckpt_path}")
            except:
                pass
            continue
    
    if not loaded:
        print("All checkpoints corrupted or failed to load. Starting from scratch...")
        model = TinyDTVi(config)
        init_from = 'scratch'

model.to(device)

# --- LOAD TOKENIZER FOR MONITORING ---
print("tokenizer path: ./tinyDTVi-tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./tinyDTVi-tokenizer")



# --- CORE VRAM OPTIMIZATION TECHNIQUES ---
# 1. Gradient Checkpointing
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()

# 2. Optimizer 8-bit
try:
    import bitsandbytes as bnb
    print("Using 8-bit AdamW...")
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{'params': decay_params, 'weight_decay': weight_decay},
                    {'params': nodecay_params, 'weight_decay': 0.0}]
    optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=(beta1, beta2))
except:
    print("bitsandbytes fallback...")
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), 'cuda')

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters: return learning_rate * it / warmup_iters
    if it > lr_decay_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# Main Training Loop
print("Training started...")
X, Y = get_batch('train')
t0 = time.time()

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        print(f"Step {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        
        # --- DISPLAY PREDICTIONS AND LABELS ---
        model.eval()
        
        # Get a real sample from Validation set
        x_samp, y_samp = get_batch('val')
        
        # 1. Ground Truth: Original content of the sample (using x_samp to get full context)
        y_ground_truth = tokenizer.decode(x_samp[0].tolist(), skip_special_tokens=True)
        
        # 2. AI Prediction: Model uses the initial prefix as a prompt and generates the rest
        # Example: taking the first 32 tokens as the "prompt"
        prompt_len = 32 if block_size > 32 else block_size // 2
        x_prompt = x_samp[0:1, :prompt_len]
        with ctx:
            y_gen_ids = model.generate(x_prompt, max_new_tokens=block_size - prompt_len, temperature=0.8, top_k=50)
        y_predict_text = tokenizer.decode(y_gen_ids[0].tolist(), skip_special_tokens=True)

        print(f"\n[GROUND TRUTH (VAL) at Step {iter_num}]:\n{y_ground_truth}\n" + "."*30)
        print(f"[AI PREDICTION (FROM VAL) at Step {iter_num}]:\n{y_predict_text}\n" + "-"*50)
        model.train()

        with open(log_file, 'a', newline='') as f:
            csv.writer(f).writerow([iter_num, losses['train'], losses['val'], lr])

        
        # Save Best
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': config.__dict__,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state()
            }
            torch.save(ckpt, os.path.join(out_dir, 'ckpt_best.pt'))

    # Training Step
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        loss.backward()
    
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Log Progress
    if iter_num % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item()*gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms, lr {lr:e}")

    # Periodic Save (Prevent data loss on crash)
    if iter_num % save_interval == 0:
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': config.__dict__,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }
        torch.save(ckpt, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
        
    iter_num += 1