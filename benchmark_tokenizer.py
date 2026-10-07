import torch
from transformers import AutoTokenizer
import tiktoken
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN) to prevent warnings when downloading models
load_dotenv()

# 1. List of Tokenizers to compare (Updated with 500M class competitors)
MODELS = {
    "tinyDTVi-500M": "./tinyDTVi-tokenizer",
    "PhoBERT-base": "vinai/phobert-base",
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", # Strongest direct competitor
    "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M", # Same SLM segment
    "Llama-3-8B (Mirror)": "unsloth/llama-3-8b-bnb-4bit", # Mirror to avoid 403 errors
    "Llama-2-7B (Mirror)": "daryl149/llama-2-7b-hf", # Original Llama-2 mirror
    "GPT-4": "gpt4", # Processed via tiktoken
    "ViGPT-2 (7B)": "bkai-foundation-models/vietnamese-llama-2-7b-120gb",
}

def get_fertility(tokenizer_name, path, texts):
    # Handle GPT-4 separately via tiktoken
    if tokenizer_name == "GPT-4":
        enc = tiktoken.get_encoding("cl100k_base")
        encode_fn = lambda x: enc.encode(x)
    else:
        # Use trust_remote_code=True for certain specific tokenizers
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        # add_special_tokens=False to measure pure text compression
        encode_fn = lambda x: tokenizer.encode(x, add_special_tokens=False)

    total_tokens = 0
    total_words = 0
    total_chars = 0
    
    for t in texts:
        words = len(t.split())
        if words == 0: continue
        
        tokens_list = encode_fn(t)
        total_words += words
        total_tokens += len(tokens_list)
        total_chars += len(t)
        
    fertility = total_tokens / total_words
    bytes_per_token = total_chars / total_tokens # Add chars per token metric for Paper
    return fertility, bytes_per_token

# 2. Benchmark Dataset (Using diverse Vietnamese sentences)
test_sentences = [
    # --- Academic & News Vietnamese ---
    "Trí tuệ nhân tạo đang tái định nghĩa cấu trúc kinh tế toàn cầu trong kỷ nguyên số.",
    "Đại học Quốc gia Hà Nội khẳng định vị thế trong bảng xếp hạng các trường đại học hàng đầu thế giới.",
    
    # --- Daily Vietnamese (CulturaX style) ---
    "Sáng nay mình đi ăn bát phở Thìn mà thấy hương vị vẫn đậm đà như ngày đầu mới ăn vậy.",
    "Bạn có khỏe không? Hôm qua mình thấy bạn có vẻ mệt mỏi khi đến lớp học của thầy Peter.",

    "Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc hằng ngày.",
    "Trường Đại học Bách khoa Hà Nội là một trong những ngôi trường kỹ thuật hàng đầu.",
    "Sáng nay, chính phủ đã ban hành nghị định mới về phát triển kinh tế số.",
    "Công thức toán học cơ bản bao gồm cộng, trừ, nhân và chia.",
    "Sầu riêng là loại trái cây đặc sản của vùng đồng bằng sông Cửu Long.", # Added VN entity

    # --- Standard English (To compare with Llama/GPT) ---
    # "Artificial Intelligence is fundamentally reshaping the global economic landscape in the digital era.",
    # "Are you okay? You looked quite unwell when you arrived at Peter's class yesterday.",
    # "This is for you; I am truly happy about it because I love you so much, brother.",

    # --- Math & Code (Check special character compression) ---
    # "The quadratic formula is defined as x = (-b ± sqrt(b^2 - 4ac)) / 2a.",
    # "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"
]

print(f"{'Model':<20} | {'Fertility':<12} | {'Chars/Token':<12}")
print("-" * 50)

results = []
for name, path in MODELS.items():
    try:
        fertility, cpt = get_fertility(name, path, test_sentences)
        print(f"{name:<20} | {fertility:.4f}      | {cpt:.2f}")
        results.append((name, fertility))
    except Exception as e:
        print(f"{name:<20} | Error: {str(e)[:30]}...")

# 3. Evaluation suggestions for Paper
if results:
    ours = [r[1] for r in results if "tinyDTVi" in r[0]][0]
    qwen = [r[1] for r in results if "Qwen" in r[0]][0]
    improvement = ((qwen - ours) / qwen) * 100
    print(f"\nNote for Arxiv: tinyDTVi compresses approximately {improvement:.2f}% better than Qwen2.5")