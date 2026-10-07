import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import gc
import time

# --- CONFIGURATION ---
INPUT_TXT = "clean_corpus.txt"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
PROGRESS_FILE = "last_position.txt" # Stores: byte_offset,line_count
TOKENIZER_PATH = "./tinyDTVi-tokenizer"

# Extract 1 line for Validation every 200 lines (0.5% - suitable for a 135GB file)
VAL_EVERY_N_LINES = 200

def stage_2_mega_file():
    # Disable low-level parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # use_fast=False to use pure Python, slower but highly stable
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    eos_id = tokenizer.eos_token_id or 0

    # Resume progress
    start_byte = 0
    line_count = 0
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                content = f.read().strip()
                if content:
                    data = content.split(',')
                    start_byte = int(data[0])
                    line_count = int(data[1]) if len(data) > 1 else 0
        except:
            pass

    print(f"Resuming at Byte: {start_byte} | Line: {line_count}")

    # Open files for appending
    f_train = open(TRAIN_BIN, "ab")
    f_val = open(VAL_BIN, "ab")

    try:
        with open(INPUT_TXT, "r", encoding="utf-8", errors="ignore") as f_in:
            f_in.seek(start_byte)
            
            pbar = tqdm(desc="Tokenizing 135GB", unit=" lines", initial=line_count)
            
            while True:
                line = f_in.readline()
                if not line: break
                
                text = line.strip()
                if text:
                    try:
                        # Single-threaded tokenization
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        ids.append(eos_id)
                        arr = np.array(ids, dtype=np.uint16)
                        
                        # Split Train/Val
                        if line_count % VAL_EVERY_N_LINES == 0:
                            f_val.write(arr.tobytes())
                        else:
                            f_train.write(arr.tobytes())
                    except Exception:
                        continue
                
                line_count += 1
                
                # Update progress bar every 500 lines
                if line_count % 500 == 0:
                    pbar.update(500)
                    # "Throttle" CPU: Sleep for 2ms to stabilize voltage
                    time.sleep(0.002)

                # Save Checkpoint every 20,000 lines to prevent data loss on crash
                if line_count % 20000 == 0:
                    curr_pos = f_in.tell()
                    with open(PROGRESS_FILE, "w") as f_prog:
                        f_prog.write(f"{curr_pos},{line_count}")
                    f_train.flush()
                    f_val.flush()
                    gc.collect()

    except Exception as e:
        print(f"\nSystem error at line {line_count}: {e}")
        # Return error code 1 for Bash script to restart
        exit(1)
    finally:
        f_train.close()
        f_val.close()
        print(f"\nFiles closed. Current progress: Line {line_count}")

if __name__ == "__main__":
    stage_2_mega_file()