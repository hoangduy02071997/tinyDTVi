import torch
from transformers import AutoTokenizer
from tinyDTVi_model import TinyDTVi, TinyDTViConfig
import os

# --- CẤU HÌNH ---
# Bạn sửa lại đường dẫn tới file weights (.pt) mới nhất của bạn ở đây nhé
CKPT_PATH = "tinyDTVi-v1-non-moe/ckpt_best.pt" # SỬA DÒNG NÀY (VD: "tinyDTVi-v1-non-moe/ckpt_best.pt")
TOKENIZER_PATH = "./tinyDTVi-tokenizer-v2"
PROMPT = "Trí tuệ nhân tạo (AI) là" # Câu mồi
MAX_NEW_TOKENS = 50 # Số từ muốn sinh ra
# ---------------

def main():
    if not os.path.exists(CKPT_PATH):
        print(f"❌ Không tìm thấy file weights tại: {CKPT_PATH}")
        print("Vui lòng sửa lại biến CKPT_PATH trong code (dòng 8) trỏ đúng tới file .pt của bạn.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Đang chạy trên: {device.upper()}")

    # 1. Load Tokenizer
    print("⏳ Đang load Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # 2. Load Checkpoint
    print(f"⏳ Đang load Checkpoint từ {CKPT_PATH}...")
    checkpoint = torch.load(CKPT_PATH, map_location=device)
    
    # 3. Khởi tạo Model
    print("⏳ Đang khởi tạo Model...")
    model_args = checkpoint.get('model_args', {})
    config = TinyDTViConfig(**model_args)
    model = TinyDTVi(config)

    # Sửa lỗi tiền tố (prefix) do quá trình train DDP sinh ra
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model đã sẵn sàng!\n")

    # 4. Chạy sinh chữ (Inference)
    input_ids = torch.tensor(tokenizer.encode(PROMPT), dtype=torch.long).unsqueeze(0).to(device)

    print(f"✍️ Mồi: '{PROMPT}'")
    print("🤖 Đang suy nghĩ...\n")

    with torch.no_grad():
        # model.generate có sẵn trong class TinyDTVi
        out_ids = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=0.8, top_k=50)

    out_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
    
    print("\n================== KẾT QUẢ ==================")
    print(out_text)
    print("=============================================\n")

if __name__ == "__main__":
    main()
