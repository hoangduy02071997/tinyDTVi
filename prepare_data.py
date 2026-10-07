import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import gc
import time

# --- CẤU HÌNH ---
INPUT_TXT = "clean_corpus.txt"
TRAIN_BIN = "train.bin"
VAL_BIN = "val.bin"
PROGRESS_FILE = "last_position.txt" # Lưu: byte_offset,line_count
TOKENIZER_PATH = "./tinyDTVi-tokenizer-v2"

# Cứ 250 dòng lấy 1 dòng cho Val (0.4% - Phù hợp file 135GB)
VAL_EVERY_N_LINES = 200

def stage_2_mega_file():
    # Tắt song song hóa tầng thấp
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # use_fast=False để dùng Python thuần, chậm nhưng cực kỳ ổn định cho i9-13th
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    eos_id = tokenizer.eos_token_id or 0

    # Khôi phục tiến trình
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

    print(f"🚀 Resume tại Byte: {start_byte} | Dòng: {line_count}")

    # Mở file ghi nối tiếp
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
                        # Tokenize đơn luồng
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        ids.append(eos_id)
                        arr = np.array(ids, dtype=np.uint16)
                        
                        # Chia Val/Train
                        if line_count % VAL_EVERY_N_LINES == 0:
                            f_val.write(arr.tobytes())
                        else:
                            f_train.write(arr.tobytes())
                    except Exception:
                        continue
                
                line_count += 1
                
                # Cập nhật thanh tiến trình mỗi 500 dòng
                if line_count % 500 == 0:
                    pbar.update(500)
                    # "Hãm phanh" CPU: Nghỉ 2ms để ổn định điện áp V-core
                    time.sleep(0.002)

                # Lưu Checkpoint mỗi 20.000 dòng để lỡ sập không mất công nhiều
                if line_count % 20000 == 0:
                    curr_pos = f_in.tell()
                    with open(PROGRESS_FILE, "w") as f_prog:
                        f_prog.write(f"{curr_pos},{line_count}")
                    f_train.flush()
                    f_val.flush()
                    gc.collect()

    except Exception as e:
        print(f"\n❌ Lỗi hệ thống tại dòng {line_count}: {e}")
        # Trả về mã lỗi 1 để file Bash biết và chạy lại
        exit(1)
    finally:
        f_train.close()
        f_val.close()
        print(f"\n✅ Đã đóng file. Tiến trình hiện tại: Dòng {line_count}")

if __name__ == "__main__":
    stage_2_mega_file()