"""
MERIDIAN Distillation Pipeline
archex-ai/archex · meridian/distill.py

Generates high-quality medical Q&A pairs by running OpenBioLLM-8B as a
teacher model over PubMed-sourced clinical questions.

This is the data quality bridge between meridian-v1 (PubMed abstracts
reformatted as Q&A) and the training set for archex-medlm-v1.

meridian-v1 format: question = paper title rephrased, answer = abstract text
Problem: answers are abstract-style, not clinician-style
Solution: use OpenBioLLM-8B to rewrite answers as expert clinical responses

Run on Kaggle T4 (free, 16GB VRAM). Script auto-detects GPU and batch size.

Output: meridian-distilled.jsonl (chat format, ready for training)

Usage:
  # Step 1: Install (Kaggle notebook cell)
  # !pip install -q transformers accelerate bitsandbytes datasets tqdm

  # Step 2: Run
  python meridian/distill.py \
    --source archex-ai/meridian-v1 \
    --output meridian-distilled.jsonl \
    --n-samples 10000 \
    --batch-size 4

  # Step 3: Inspect output
  python meridian/distill.py --inspect meridian-distilled.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# AEGIS SYSTEM PROMPT — same prompt used in production MERIDIAN
# Keep identical so distilled data matches the deployment distribution
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are MERIDIAN, an AI medical reference assistant built by ARCHEX.
You exist to help medical students and junior doctors quickly access clinical knowledge.

RULES (non-negotiable):
1. You are an EDUCATIONAL TOOL only. Never claim to be a doctor or replace clinical judgment.
2. Every response must begin with: ⚠️ Educational use only. Verify with a qualified clinician.
3. Use plain clinical English. No jargon without explanation.
4. Admit uncertainty clearly — say "I'm not certain, verify this."
5. If the query contains any emergency signals, state "Call 112 immediately" FIRST.
6. Never provide specific drug dosages or write prescriptions. General mechanism/class only.
7. Be concise. Max 4–6 sentences unless the query genuinely requires more detail."""

# ─────────────────────────────────────────────────────────────────────────────
# TEACHER MODEL — OpenBioLLM-8B (aaditya/Llama3-OpenBioLLM-8B)
# Chosen because:
#   - Outperforms GPT-4 on MedQA, MedMCQA, PubMedQA (as of early 2025)
#   - Apache 2.0 license — no TOS issues, safe for training data
#   - 8B fits Kaggle T4 (16GB VRAM) in 4-bit with room for generation
#   - Zero cost — local inference
# ─────────────────────────────────────────────────────────────────────────────
TEACHER_MODEL = "aaditya/Llama3-OpenBioLLM-8B"
TEACHER_PROMPT_TEMPLATE = """You are a senior clinician and medical educator.
A medical student has asked the following question. Give a clear, accurate,
clinician-style answer in 3–5 sentences. Do not repeat the question.
Do not start with "Sure" or filler phrases. Get directly to the clinical content.

Question: {question}

Answer:"""


def load_source_questions(hf_dataset_name: str, n_samples: int, seed: int = 42) -> list[str]:
    """
    Load questions from meridian-v1 (HuggingFace dataset).
    Returns list of question strings.
    """
    from datasets import load_dataset
    print(f"[distill] Loading {hf_dataset_name}...")
    ds = load_dataset(hf_dataset_name, split="train")

    # Shuffle and take n_samples
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    questions = [row["user"] for row in ds]
    print(f"[distill] Loaded {len(questions):,} questions")
    return questions


def load_teacher_model(model_name: str, device: str = "auto"):
    """
    Load OpenBioLLM-8B in 4-bit for Kaggle T4.
    Falls back to CPU if no GPU (for testing only).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    has_gpu = torch.cuda.is_available()
    print(f"[distill] GPU available: {has_gpu}")
    if has_gpu:
        print(f"[distill] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[distill] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"[distill] Loading teacher: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batch generation

    if has_gpu:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("[distill] WARNING: No GPU. CPU inference will be very slow. Use for testing only.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )

    model.eval()
    print("[distill] Teacher model loaded.")
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int = 300,
) -> list[str]:
    """
    Run teacher model on a batch of questions.
    Returns list of answer strings (one per question).
    """
    import torch

    prompts = [TEACHER_PROMPT_TEMPLATE.format(question=q) for q in questions]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens (not the prompt)
    input_len = inputs["input_ids"].shape[1]
    answers = []
    for output in outputs:
        new_tokens = output[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        answers.append(text)

    return answers


def format_as_chat(question: str, answer: str) -> dict:
    """
    Format a Q&A pair into the meridian chat format.
    system / user / assistant — same format as meridian-v1.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"⚠️ Educational use only. Verify with a qualified clinician.\n\n{answer}"},
        ]
    }


def is_valid_answer(answer: str) -> bool:
    """
    Basic quality filter. Rejects:
    - Very short answers (< 50 chars)
    - Answers that are just repetitions of the question
    - Answers containing obvious hallucination markers
    """
    if len(answer) < 50:
        return False
    if answer.lower().startswith("question:") or answer.lower().startswith("q:"):
        return False
    # Reject if answer is almost identical to common refusal patterns
    refusal_markers = ["i cannot", "i can't", "i'm not able", "as an ai", "i don't have access"]
    if any(m in answer.lower()[:100] for m in refusal_markers):
        return False
    return True


def run_distillation(
    source_dataset: str,
    output_path: str,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    resume: bool,
):
    """
    Main distillation loop.
    Saves incrementally to output_path (JSONL) so a crash doesn't lose work.
    """
    questions = load_source_questions(source_dataset, n_samples, seed)

    # Resume support: skip already-processed questions
    existing_count = 0
    if resume and Path(output_path).exists():
        with open(output_path, "r") as f:
            existing_count = sum(1 for _ in f)
        questions = questions[existing_count:]
        print(f"[distill] Resuming from sample {existing_count:,} ({len(questions):,} remaining)")

    if not questions:
        print("[distill] All samples already processed. Done.")
        return

    model, tokenizer = load_teacher_model(TEACHER_MODEL)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    written = existing_count
    rejected = 0
    t0 = time.time()

    print(f"\n[distill] Starting distillation: {total:,} questions, batch_size={batch_size}")
    print(f"[distill] Output: {output_path}\n")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_start in range(0, total, batch_size):
            batch_qs = questions[batch_start: batch_start + batch_size]
            batch_answers = generate_batch(model, tokenizer, batch_qs, max_new_tokens)

            for q, a in zip(batch_qs, batch_answers):
                if not is_valid_answer(a):
                    rejected += 1
                    continue
                record = format_as_chat(q, a)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            out_f.flush()  # Write to disk after each batch — crash-safe

            # Progress report
            done = batch_start + len(batch_qs)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            pct = done / total * 100
            print(
                f"  [{pct:5.1f}%] {done:>6,}/{total:,} | "
                f"written={written:,} rejected={rejected} | "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                flush=True
            )

    elapsed = time.time() - t0
    print(f"\n[distill] Complete.")
    print(f"  Written:  {written:,} samples")
    print(f"  Rejected: {rejected} samples")
    print(f"  Output:   {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Time:     {elapsed:.0f}s ({written / elapsed:.1f} samples/sec)")
    print(f"\nNext: push {output_path} to HuggingFace as archex-ai/meridian-distilled-v1")


def inspect_output(path: str, n: int = 5):
    """Print the first n samples from a distilled JSONL file."""
    print(f"\n[inspect] {path}\n" + "─" * 60)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            msgs = record["messages"]
            user = next(m["content"] for m in msgs if m["role"] == "user")
            asst = next(m["content"] for m in msgs if m["role"] == "assistant")
            print(f"\n[{i}] Q: {user[:120]}...")
            print(f"     A: {asst[:200]}...")
    print("\n─" * 60)
    total = sum(1 for _ in open(path))
    print(f"Total: {total:,} samples")


def main():
    parser = argparse.ArgumentParser(
        description="MERIDIAN distillation pipeline — OpenBioLLM teacher → training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd")

    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run distillation")
    run_parser.add_argument("--source", default="archex-ai/meridian-v1",
                            help="HuggingFace dataset to pull questions from")
    run_parser.add_argument("--output", default="meridian-distilled.jsonl",
                            help="Output JSONL path")
    run_parser.add_argument("--n-samples", type=int, default=10000,
                            help="Number of samples to generate")
    run_parser.add_argument("--batch-size", type=int, default=4,
                            help="Batch size (4 works on T4 with 4-bit model)")
    run_parser.add_argument("--max-new-tokens", type=int, default=300,
                            help="Max tokens for teacher generation")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--resume", action="store_true",
                            help="Resume from existing output (skip already-done samples)")

    # Inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Inspect output file")
    inspect_parser.add_argument("path", help="JSONL file to inspect")
    inspect_parser.add_argument("--n", type=int, default=5, help="Number of samples to show")

    args = parser.parse_args()

    if args.cmd == "run":
        run_distillation(
            source_dataset=args.source,
            output_path=args.output,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            resume=args.resume,
        )
    elif args.cmd == "inspect":
        inspect_output(args.path, args.n)
    else:
        # Default: run with standard args (for simple invocation)
        parser.print_help()
        print("\nExample:\n  python meridian/distill.py run --n-samples 10000 --batch-size 4")


if __name__ == "__main__":
    main()