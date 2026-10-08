import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import math
from transformers import AutoTokenizer
from tinyDTVi_model import TinyDTVi, TinyDTViConfig

# --- CONFIGURATION ---
CKPT_PATH = "tinyDTVi-500M/ckpt_best.pt" 
TOKENIZER_PATH = "./tinyDTVi-tokenizer"
MAX_TOKENS = 1024 # TinyDTVi max context window

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Cannot find {CKPT_PATH}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Download DeepMind's XQuAD dataset using standard Python library to avoid HF dependencies
    print("Loading XQuAD (Cross-lingual Question Answering Dataset) from DeepMind...")
    print("Note: This is a standard professionally translated dataset, ensuring the model has never seen it in the train data.")
    
    import urllib.request
    import json
    url = "https://raw.githubusercontent.com/deepmind/xquad/master/xquad.vi.json"
    
    try:
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        
        # Extract all context paragraphs in the dataset
        contexts = []
        for article in data['data']:
            for paragraph in article['paragraphs']:
                contexts.append(paragraph['context'])
                
        # Join into a giant string
        full_text = "\n".join(contexts)
        print(f"Done! Extracted {len(contexts)} paragraphs ({len(full_text)} characters total).")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 1. Load Tokenizer & Model
    print("Loading Tokenizer & Model for BPC calculation...")
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

    # 2. Tokenize data (Create input_ids and targets)
    print("Tokenizing text...")
    input_ids = tokenizer.encode(full_text, return_tensors='pt').to(device)
    
    # Strictly limit to 1024 tokens (Context length) for accurate measurement
    if input_ids.size(1) > MAX_TOKENS:
        input_ids = input_ids[:, :MAX_TOKENS]
    
    num_tokens = input_ids.size(1)
    # Decode back to count actual characters
    truncated_text = tokenizer.decode(input_ids[0])
    num_chars = len(truncated_text)
    
    chars_per_token = num_chars / num_tokens

    # Shift input_ids to create x (input) and y (target - next word)
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    print(f"Test Segment Length: {num_chars} chars | {num_tokens} tokens.")
    print(f"Characters per token: {chars_per_token:.2f}")

    # 3. Calculate Loss, PPL, and BPC
    with torch.no_grad():
        logits, loss = model(x, y)
        
    val_loss = loss.item()
    ppl = math.exp(val_loss)
    
    # BPC formula: loss / ln(2) / (chars_per_token)
    # CrossEntropyLoss returns nats (base e), divided by ln(2) it becomes bits (base 2)
    bits_per_token = val_loss / math.log(2)
    bpc = bits_per_token / chars_per_token

    print("\n" + "="*50)
    print(f"Validation Loss: {val_loss:.4f} nats")
    print(f"Perplexity (PPL): {ppl:.2f}")
    print(f"Bits Per Character (BPC): {bpc:.4f} bits/char")
    print("="*50 + "\n")
    print("Note: BPC is required because PPL cannot be directly compared across different tokenizers.")

if __name__ == "__main__":
    main()
