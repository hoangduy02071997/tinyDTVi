import os
import random
import gc
from datasets import load_from_disk
from tqdm import tqdm
from clean_utils import clean_text # Ensure clean_utils.py is in the same directory

# --- CONFIGURATION ---
DATA_DIR = "./data"
OUTPUT_TXT = "clean_corpus.txt"
SEED = 42

def get_int_hash(text):
    # Use Python's built-in hash (returns an integer) for memory efficiency
    return hash(text[:500]) 

def stage_1():
    print("STAGE 1: Extraction, Cleaning & Deduplication")
    
    # 1. Load datasets
    print("Loading datasets from disk...")
    wiki = load_from_disk(os.path.join(DATA_DIR, "vi_wiki"))
    vnews = load_from_disk(os.path.join(DATA_DIR, "vnews"))
    culturax = load_from_disk(os.path.join(DATA_DIR, "culturax_vi_200p"))
    
    # 2. Create interleaving indices
    print("Shuffling dataset order...")
    indices = []
    indices.extend([('w', i) for i in range(len(wiki))])
    indices.extend([('v', i) for i in range(len(vnews))])
    indices.extend([('c', i) for i in range(len(culturax))])
    
    random.seed(SEED)
    random.shuffle(indices)
    
    # 3. Process and write
    seen_hashes = set()
    total = len(indices)
    
    print(f"Processing {total:,} samples...")
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        pbar = tqdm(total=total, desc="Extracting")
        for tag, idx in indices:
            try:
                # Get raw text
                if tag == 'w': item = wiki[idx]
                elif tag == 'v': item = vnews[idx]
                else: item = culturax[idx]
                
                raw_text = item['text']
                if not raw_text or len(raw_text) < 150: 
                    pbar.update(1)
                    continue
                
                # Clean text
                cleaned = clean_text(raw_text[:10000])
                if not cleaned or len(cleaned) < 150:
                    pbar.update(1)
                    continue
                
                # Deduplicate
                h = get_int_hash(cleaned)
                if h in seen_hashes:
                    pbar.update(1)
                    continue
                
                # Protect RAM (Set of ~15M ints is fine with 32GB RAM)
                if len(seen_hashes) < 15_000_000:
                    seen_hashes.add(h)
                
                # Write to file (one document per line, replace inner newlines)
                f.write(cleaned.replace("\n", " ") + "\n")
                
            except Exception:
                continue
            finally:
                pbar.update(1)
                if pbar.n % 100000 == 0:
                    gc.collect()

    print(f"\nCompleted Stage 1! Clean file: {OUTPUT_TXT}")

if __name__ == "__main__":
    stage_1()