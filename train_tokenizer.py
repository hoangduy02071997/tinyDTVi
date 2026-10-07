import os
import gc
from tokenizers import Tokenizer, normalizers, pre_tokenizers, models, trainers, decoders
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from clean_utils import clean_text

# --- CẤU HÌNH ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
VOCAB_SIZE = 50304
OUTPUT_DIR = "tinyDTVi-tokenizer"
DATA_DIR = "./data" 
TEMP_CORPUS_FILE = os.path.join(DATA_DIR, "train_corpus.txt")

def prepare_corpus_file(data_dir, output_file):
    """Trích xuất toàn bộ dữ liệu ra file txt thô."""
    dataset_names = ["vi_wiki", "vnews", "culturax_vi_200p"]
    
    # Kiểm tra nếu file đã tồn tại thì không cần trích xuất lại để tiết kiệm thời gian
    if os.path.exists(output_file):
        print(f"ℹ️ File {output_file} đã tồn tại, bỏ qua bước trích xuất.")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for name in dataset_names:
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                print(f"⚠️ Cảnh báo: Không tìm thấy {path}, bỏ qua...")
                continue
                
            print(f"📖 Đang trích xuất dữ liệu từ: {name}...")
            ds = load_from_disk(path)
            
            count = 0
            batch_size = 1000
            for i in range(0, len(ds), batch_size):
                batch = ds[i : i + batch_size]["text"]
                for t in batch:
                    if t and len(t) > 10:
                        truncated_text = t[:10000] 
                        cleaned_val = clean_text(truncated_text)
                        if cleaned_val:
                            # Ghi mỗi doc vào 1 dòng
                            f.write(cleaned_val.replace("\n", " ") + "\n")
                            count += 1
                
                if count % 100000 == 0:
                    gc.collect()
            
            print(f"✅ Đã xong {name}: Tổng {count:,} dòng.")
            del ds
            gc.collect()

def batch_iterator(file_path, batch_size=1000):
    """Đọc file cực lớn theo từng batch để không bị Segmentation Fault."""
    with open(file_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

def main():
    # 1. Chuẩn bị file corpus 
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    prepare_corpus_file(DATA_DIR, TEMP_CORPUS_FILE)

    # 2. Khởi tạo Tokenizer Byte-Level BPE (Chuẩn ArXiv)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()]) 
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 3. Cấu hình Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask|"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2 
    )

    # 4. Huấn luyện bằng Iterator (Giải pháp cho file 135GB)
    print(f"⏳ Bắt đầu huấn luyện Tokenizer (Streaming từ ổ cứng)...")
    # Thay vì truyền file list, ta truyền generator để tiết kiệm RAM
    tokenizer.train_from_iterator(batch_iterator(TEMP_CORPUS_FILE), trainer)

    # 5. Lưu kết quả
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        mask_token="<mask|"
    )
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"🎉 Hoàn thành! Tokenizer lưu tại: {OUTPUT_DIR}")
    
    # 6. Dọn dẹp file tạm (Tùy chọn)
    # print(f"🧹 Đang xóa file tạm {TEMP_CORPUS_FILE}...")
    # os.remove(TEMP_CORPUS_FILE)

if __name__ == "__main__":
    main()