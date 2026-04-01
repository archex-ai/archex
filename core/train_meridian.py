"""
MERIDIAN Phase 3 — LoRA Fine-Tuning Script
Target: Kaggle T4 (single GPU, 16GB VRAM)
Base model: mistralai/Mistral-7B-Instruct-v0.3
Trigger: run when clean_samples >= 100 in train.csv

Founder: Md Kaif Ali Khan | ARCHEX
"""

# ─── INSTALL (run this cell first on Kaggle) ───
# !pip install -q transformers peft datasets accelerate bitsandbytes trl

import os
import json
import csv
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./meridian-lora"
TRAIN_CSV = "train.csv"       # output of Phase 2.5 filter
HF_DATASET = "archex-ai/meridian-v1"
MIN_SAMPLES = 100
MAX_SEQ_LEN = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 3

SYSTEM_PROMPT = """You are MERIDIAN, an AI medical reference assistant built by ARCHEX.
You are an EDUCATIONAL TOOL only. Never claim to be a doctor or replace clinical judgment.
Every response must begin with: ⚠️ Educational use only. Verify with a qualified clinician.
Use plain clinical English. Admit uncertainty clearly.
If the query contains emergency signals, state 'Call 112 immediately' FIRST.
Never provide specific drug dosages or prescriptions. General mechanism/class only.
Be concise."""

# ─────────────────────────────────────────────
# STEP 1: LOAD + MERGE DATASETS
# ─────────────────────────────────────────────
def load_train_csv(path: str) -> list[dict]:
    """Load Phase 2.5 filtered CSV → list of {question, answer} dicts."""
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Skipping local CSV.")
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if q and a and len(q) > 10 and len(a) > 20:
                rows.append({"question": q, "answer": a})
    print(f"[INFO] Loaded {len(rows)} rows from {path}")
    return rows

def load_hf_dataset() -> list[dict]:
    """Load meridian-v1 from HuggingFace. Format: system/user/assistant."""
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, split="train")
    rows = []
    for item in ds:
        # meridian-v1 is chat format: messages list
        messages = item.get("messages", [])
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        if user_msg and asst_msg:
            rows.append({"question": user_msg.strip(), "answer": asst_msg.strip()})
    print(f"[INFO] Loaded {len(rows)} rows from HuggingFace dataset")
    return rows

def build_merged_dataset() -> Dataset:
    """Merge CSV logs + HF dataset. Deduplicate by question."""
    csv_rows = load_train_csv(TRAIN_CSV)
    hf_rows = load_hf_dataset()
    all_rows = hf_rows + csv_rows  # HF base first, real user queries appended

    # Deduplicate
    seen = set()
    deduped = []
    for row in all_rows:
        key = row["question"][:80].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    print(f"[INFO] Total unique samples after merge: {len(deduped)}")

    if len(csv_rows) < MIN_SAMPLES:
        print(f"[WARN] Only {len(csv_rows)} clean user samples. "
              f"Minimum recommended: {MIN_SAMPLES}. Proceeding with HF data only.")

    return Dataset.from_list(deduped)

# ─────────────────────────────────────────────
# STEP 2: FORMAT TO CHAT TEMPLATE
# ─────────────────────────────────────────────
def format_sample(sample: dict, tokenizer) -> dict:
    """
    Convert {question, answer} → Mistral chat template string.
    Uses tokenizer.apply_chat_template for correctness.
    NOT alpaca format — chat format only.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    # apply_chat_template handles BOS/EOS/special tokens correctly
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# ─────────────────────────────────────────────
# STEP 3: LOAD MODEL (4-bit quantized for T4)
# ─────────────────────────────────────────────
def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for SFT

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    print(f"[INFO] Model loaded: {BASE_MODEL}")
    return model, tokenizer

# ─────────────────────────────────────────────
# STEP 4: APPLY LORA
# ─────────────────────────────────────────────
def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# ─────────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────────
def train(model, tokenizer, dataset: Dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=25,
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",  # Set to "wandb" if tracking
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=training_args,
        packing=False,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[INFO] LoRA adapter saved to {OUTPUT_DIR}")
    return trainer

# ─────────────────────────────────────────────
# STEP 6: EVALUATION — master vs slave
# ─────────────────────────────────────────────
def evaluate_slave(model, tokenizer, test_queries: list[str]):
    """
    Run a quick qualitative eval: print slave responses for test queries.
    Compare mentally against Groq/master responses.
    Quantitative BLEU/ROUGE eval is future work.
    """
    from peft import PeftModel
    model.eval()
    print("\n" + "="*60)
    print("SLAVE MODEL EVALUATION")
    print("="*60)
    for query in test_queries:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {query[:80]}")
        print(f"A: {decoded.strip()[:300]}")
    print("="*60)
    print("If slave responses are clinically coherent and safe → mark for promotion.")
    print(f"Upload {OUTPUT_DIR}/ to HuggingFace as archex-ai/meridian-slave-v0.3")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("[ARCHEX] MERIDIAN Phase 3 Training Pipeline Starting...")

    # Sanity check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected. Run this on Kaggle T4.")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Build dataset
    dataset = build_merged_dataset()
    print(f"[INFO] Final dataset size: {len(dataset)} samples")

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Format dataset
    dataset = dataset.map(lambda x: format_sample(x, tokenizer), remove_columns=["question", "answer"])

    # Apply LoRA
    model = apply_lora(model)

    # Train
    trainer = train(model, tokenizer, dataset)

    # Eval on sample queries
    test_queries = [
        "What are the classic features of Horner syndrome?",
        "Mechanism of action of beta blockers in heart failure?",
        "Differentials for painless jaundice in a 60-year-old?",
        "What is the role of troponin in diagnosing ACS?",
    ]
    evaluate_slave(model, tokenizer, test_queries)

    print("\n[DONE] Phase 3 complete. Review eval output before promoting slave → master.")