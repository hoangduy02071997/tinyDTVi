import torch
import math
from transformers import AutoTokenizer
from tinyDTVi_model import TinyDTVi, TinyDTViConfig
import os

CKPT_PATH = "tinyDTVi-500M/ckpt_best.pt" 
TOKENIZER_PATH = "./tinyDTVi-tokenizer"

# Load Wikipedia held-out sample via HuggingFace
print("Loading Wikipedia (Held-out) dataset...")
try:
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    wiki_texts = []
    # Skip first 1000 to avoid train-set contamination if possible, grab next 10 articles
    iterator = iter(ds)
    for _ in range(1000): next(iterator)
    for _ in range(10):
        wiki_texts.append(next(iterator)["text"])
    EVAL_TEXT = "\n".join(wiki_texts)[:10000] # Use a large 10k char chunk
    print("Successfully loaded Wikipedia sample.")
except Exception as e:
    print(f"Error loading Wiki: {e}. Please install datasets library.")
    exit(1)

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Cannot find {CKPT_PATH}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Tokenizer & Model
    print("Loading Tokenizer ⏳ Đang nạp Tokenizer & Model để tính Perplexity... Model for Perplexity calculation...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    
    config = TinyDTViConfig(**checkpoint.get('model_args', {}))
    model = TinyDTVi(config)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    import sys
    # Ensure clean_utils is accessible
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from clean_utils import clean_text
    EVAL_TEXT = clean_text(EVAL_TEXT)

    # 2. Tokenize data (Create input_ids and targets)
    input_ids = tokenizer.encode(EVAL_TEXT, return_tensors='pt').to(device)
    
    # Strictly limit to 1024 tokens to avoid RoPE crash
    MAX_TOKENS = 1024
    if input_ids.size(1) > MAX_TOKENS:
        input_ids = input_ids[:, :MAX_TOKENS]
    
    # Shift input_ids to create x (input) and y (target)
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    print(f"Test sequence length: {x.size(1)} tokens.")

    # 3. Calculate Loss 3. Tính Loss & Perplexity Perplexity
    with torch.no_grad():
        logits, loss = model(x, y)
        
    ppl = math.exp(loss.item())

    print("\n" + "="*50)
    print(f"Validation Loss: {loss.item():.4f}")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print("="*50 + "\n")
    print("Note: Lower PPL is better. Base 500M LLMs usually have PPL between 15 and 30.")

if __name__ == "__main__":
    main()
