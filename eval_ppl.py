import torch
import math
from transformers import AutoTokenizer
from tinyDTVi_model import TinyDTVi, TinyDTViConfig
import os

# --- CONFIGURATION ---
CKPT_PATH = "tinyDTVi-500M/ckpt_best.pt" 
TOKENIZER_PATH = "./tinyDTVi-tokenizer"

# Held-out sample text representing Wikipedia to measure Perplexity
# You can replace this with any long text of about 500-1000 words.
EVAL_TEXT = """Trí tuệ nhân tạo (tiếng Anh: artificial intelligence, hay máy tính thông minh) là trí tuệ được biểu diễn bởi bất kỳ một hệ thống nhân tạo nào. Thuật ngữ này thường dùng để nói đến các máy tính có mục đích không nhất định và ngành khoa học nghiên cứu về các lý thuyết và ứng dụng của trí tuệ nhân tạo.
Mặc dù trí tuệ nhân tạo có ý nghĩa rộng như là trí thông minh trong các tác phẩm khoa học viễn tưởng, nó là một trong những ngành trọng yếu của tin học. Trí tuệ nhân tạo liên quan đến cách cư xử, sự học hỏi và khả năng thích ứng thông minh của máy móc. 
Nghiên cứu về trí tuệ nhân tạo nhằm mục đích tạo ra các hệ thống máy tính có khả năng thực hiện các công việc đòi hỏi trí thông minh của con người. Các ví dụ bao gồm nhận dạng giọng nói, thị giác máy tính, dịch ngôn ngữ tự nhiên và ra quyết định chuyên gia."""
# ---------------

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Could not find {CKPT_PATH}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Tokenizer & Model
    print("Loading Tokenizer & Model to calculate Perplexity...")
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
    input_ids = tokenizer.encode(EVAL_TEXT, return_tensors='pt').to(device)
    
    # Shift input_ids to create x (input) and y (target - next token)
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    print(f"Test segment length: {x.size(1)} tokens.")

    # 3. Calculate Loss & Perplexity
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
