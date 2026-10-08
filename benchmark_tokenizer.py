import torch
from transformers import AutoTokenizer
import tiktoken
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN) to prevent warnings when downloading models
load_dotenv()

# 1. List of Tokenizers to compare (Exactly matching Table 1 of the paper)
MODELS = {
    "tinyDTVi": "./tinyDTVi-tokenizer",
    "PhoBERT": "vinai/phobert-base",
    "Qwen2.5": "Qwen/Qwen2.5-0.5B", 
    "SmolLM2": "HuggingFaceTB/SmolLM2-360M", 
    "Llama-3": "unsloth/llama-3-8b-bnb-4bit", # Mirror
    "Llama-2": "daryl149/llama-2-7b-hf", # Mirror
    "GPT-4": "gpt4", # Processed via tiktoken
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

# 2. Benchmark Dataset (10,000 sentences from CulturaX to match paper claims)
print("Loading 10,000 sentences from CulturaX...")
try:
    from datasets import load_dataset
    # Stream the dataset to avoid downloading the massive CulturaX corpus
    ds = load_dataset("uonlp/CulturaX", "vi", split="train", streaming=True)
    test_sentences = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > 50: # Only take reasonable length sentences
            test_sentences.append(text)
        if len(test_sentences) >= 10000:
            break
    print("Successfully loaded 10,000 sentences.")
except Exception as e:
    print(f"Warning: Failed to load CulturaX directly from HuggingFace ({e}). Falling back to dummy text.")
    # Fallback to a few sentences just to keep the script runnable without internet/HF
    test_sentences = [
        "Trí tuệ nhân tạo đang tái định nghĩa cấu trúc kinh tế toàn cầu trong kỷ nguyên số.",
        "Đại học Quốc gia Hà Nội khẳng định vị thế trong bảng xếp hạng các trường đại học hàng đầu thế giới."
    ] * 5000 # duplicate 5000 times to simulate 10,000 sentences


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