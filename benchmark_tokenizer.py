import os
import sys
import subprocess
import json

# Disable bytecode writing to prevent concurrent .pyc corruption
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# CRITICAL FIX: Disable tqdm background monitor thread which causes SIGSEGV race conditions with C-extensions
import tqdm
tqdm.tqdm.monitor_interval = 0

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local tinyDTVi tokenizer folder in release repository
TINYDTVI_TOKENIZER_PATH = os.path.join(SCRIPT_DIR, "tinyDTVi-tokenizer")

# 1. List of Tokenizers to compare (Table 1 of the paper)
MODELS = {
    "tinyDTVi": TINYDTVI_TOKENIZER_PATH,
    "PhoBERT": "vinai/phobert-base",
    "Qwen2.5": "Qwen/Qwen2.5-0.5B", 
    "SmolLM2": "HuggingFaceTB/SmolLM2-360M", 
    "Llama-3": "unsloth/llama-3-8b-bnb-4bit", # Mirror
    "Llama-2": "daryl149/llama-2-7b-hf", # Mirror
    "GPT-4": "gpt4", # Processed via tiktoken
}

# --- WORKER PROCESS ---
if len(sys.argv) >= 5 and sys.argv[1] == "--worker":
    name = sys.argv[2]
    path = sys.argv[3]
    dataset_file = sys.argv[4]

    if name == "GPT-4":
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        encode_fn = lambda x: enc.encode(x)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        encode_fn = lambda x: tok.encode(x, add_special_tokens=False)

    total_words = 0
    total_tokens = 0
    total_chars = 0
    count = 0

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) <= 50:
                continue
            w = len(line.split())
            if w == 0:
                continue
            toks_len = len(encode_fn(line))
            total_words += w
            total_tokens += toks_len
            total_chars += len(line)
            count += 1
            if count % 2000 == 0:
                print(f"  [{name}] {count}/10000 documents evaluated...", file=sys.stderr, flush=True)
            if count >= 10000:
                break

    fertility = total_tokens / total_words if total_words > 0 else 0
    cpt = total_chars / total_tokens if total_tokens > 0 else 0
    print(f"__RESULT__:{json.dumps({'fertility': fertility, 'cpt': cpt})}")
    os._exit(0)

# --- MAIN CONTROLLER PROCESS ---
# 2. Benchmark Dataset (10,000 documents from CulturaX-vi through whitelist clean_text)
NUM_SAMPLES = 10000
culturax_benchmark_file = os.path.join(SCRIPT_DIR, "culturax_10k_test.txt")
if not os.path.exists(culturax_benchmark_file):
    parent_file = os.path.join(SCRIPT_DIR, "..", "culturax_10k_test.txt")
    if os.path.exists(parent_file):
        culturax_benchmark_file = parent_file

if not os.path.exists(culturax_benchmark_file):
    culturax_benchmark_file = os.path.join(SCRIPT_DIR, "culturax_10k_test.txt")
    print(f"File {culturax_benchmark_file} not found. Preparing {NUM_SAMPLES:,} documents from CulturaX-vi...")
    extractor_code = f"""
import os
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
import sys
sys.path.append(r'{SCRIPT_DIR}')
from clean_utils import clean_text

output_file = r'{culturax_benchmark_file}'
print('Streaming {NUM_SAMPLES:,} documents from CulturaX-vi with whitelist clean_text...', flush=True)
ds = load_dataset('uonlp/CulturaX', 'vi', split='train', streaming=True)
count = 0
with open(output_file, 'w', encoding='utf-8') as f:
    for row in ds:
        cleaned = clean_text(row['text'])
        if len(cleaned) > 50:
            f.write(cleaned.replace('\\n', ' ') + '\\n')
            count += 1
            if count % 2000 == 0:
                print(f'Collected {{count}}/{NUM_SAMPLES:,} documents...', flush=True)
            if count >= {NUM_SAMPLES}:
                break
print(f'Saved {{count:,}} documents to {{output_file}}.', flush=True)
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", extractor_code], check=True)

print(f"Loading {NUM_SAMPLES:,} documents from {culturax_benchmark_file}...")
with open(culturax_benchmark_file, "r", encoding="utf-8") as f:
    valid_count = sum(1 for line in f if len(line.strip()) > 50)
print(f"Successfully loaded {min(valid_count, NUM_SAMPLES):,} cleaned CulturaX documents (whitelist applied).\n")

print(f"{'Model':<20} | {'Fertility':<12} | {'Chars/Token':<12}")
print("-" * 50)

results = []
this_script = os.path.abspath(__file__)

for name, path in MODELS.items():
    proc = subprocess.run(
        [sys.executable, this_script, "--worker", name, path, culturax_benchmark_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        print(f"{name:<20} | Process Error (exit code {proc.returncode})")
        if proc.stderr:
            err_lines = [l for l in proc.stderr.splitlines() if l.strip()]
            for el in err_lines[-3:]:
                print(f"    [err] {el}")
        continue

    parsed = False
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__:"):
            data = json.loads(line[len("__RESULT__:"):])
            fertility = data["fertility"]
            cpt = data["cpt"]
            print(f"{name:<20} | {fertility:.4f}      | {cpt:.2f}")
            results.append((name, fertility))
            parsed = True
            break
    if not parsed:
        print(f"{name:<20} | Output Parse Error")

# 3. Evaluation summary for Paper Table 1 & Arxiv
tinydtvi_results = [r[1] for r in results if "tinyDTVi" in r[0]]
qwen_results = [r[1] for r in results if "Qwen" in r[0]]
if tinydtvi_results and qwen_results:
    ours = tinydtvi_results[0]
    qwen = qwen_results[0]
    improvement = ((qwen - ours) / qwen) * 100
    print(f"\nNote for Arxiv: tinyDTVi compresses approximately {improvement:.2f}% better than Qwen2.5")

os._exit(0)