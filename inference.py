import torch
from transformers import AutoTokenizer
from tinyDTVi_model import TinyDTVi, TinyDTViConfig
import os

# --- CONFIGURATION ---
# Please update this path to your latest weights (.pt) file
CKPT_PATH = "tinyDTVi-500M/ckpt_best.pt" # UPDATE THIS LINE (e.g., "tinyDTVi-500M/ckpt_best.pt")
TOKENIZER_PATH = "./tinyDTVi-tokenizer"
PROMPT = "Trí tuệ nhân tạo (AI) là" # Initial prompt
MAX_NEW_TOKENS = 50 # Number of tokens to generate
# ---------------

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Weights file not found at: {CKPT_PATH}")
        print("Please update the CKPT_PATH variable in the code (line 8) to point to your .pt file.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device.upper()}")

    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 2. Load Checkpoint
    print(f"Loading Checkpoint from {CKPT_PATH}...")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    
    # 3. Initialize Model
    print("Initializing Model...")
    model_args = checkpoint.get('model_args', {})
    config = TinyDTViConfig(**model_args)
    model = TinyDTVi(config)

    # Fix state_dict prefix issue caused by DDP training
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model is ready!\n")

    # 4. Run Inference
    input_ids = torch.tensor(tokenizer.encode(PROMPT), dtype=torch.long).unsqueeze(0).to(device)

    print(f"Prompt: '{PROMPT}'")
    print("Generating...\n")

    with torch.no_grad():
        # model.generate is available in the TinyDTVi class
        out_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_k=50)

    out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    
    print("\n================== RESULT ==================")
    print(out_text)
    print("=============================================\n")

if __name__ == "__main__":
    main()
