#!/usr/bin/env python3
"""Matched direct-vs-CoT SQL experiment (gretelai/synthetic_text_to_sql) — train AND eval.

The POSITIVE CONTROL for the CoT-as-compositional-prior hypothesis. SQL composes
(joins/sub-queries) so we predict CoT HELPS — the opposite of Cypher/SPARQL.
If CoT does NOT help SQL in our own matched pipeline, the theory is dead.

Clean matched design: BOTH variants train on the SAME instances (rows with a
valid reasoning trace); the ONLY difference is the training target.
Schema is provided in the prompt (mirrors the Cypher setup).

Usage (called by drac_sql.sh):
    python drac_train_sql.py --variant direct --train-data traces.jsonl \\
        --output-dir ~/scratch/sql_direct_adapter --hf-cache $HF_CACHE
    python drac_train_sql.py --eval --variant cot --test-data test.jsonl \\
        --adapter-path ~/scratch/sql_cot_adapter/final \\
        --output-dir ~/scratch/results_sql --hf-cache $HF_CACHE
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch

BASE_MODEL = "google/gemma-2-9b-it"

SQL_BASELINE_INSTRUCTION = (
    "Generate a SQL query to answer the question.\n"
    "Use only the tables and columns in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n"
    "SQL output:"
)
SQL_COT_INSTRUCTION = (
    "Generate a SQL query to answer the question.\n"
    "Use only the tables and columns in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n\n"
    "Think step by step, then provide the SQL query."
)


class CompletionOnlyCollator:
    """Mask loss before the assistant response template (same as Gemma baselines)."""

    def __init__(self, tokenizer, response_template: str, max_length: int):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
        self.max_length = max_length

    def _build_labels(self, input_ids, attention_mask):
        labels = input_ids.clone()
        rt, rt_len = self.response_template_ids, len(self.response_template_ids)
        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()
            start = None
            for j in range(len(ids_list) - rt_len + 1):
                if ids_list[j:j + rt_len] == rt:
                    start = j + rt_len
                    break
            if start is None:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100
        labels[attention_mask == 0] = -100
        return labels

    def __call__(self, features):
        if features and "input_ids" in features[0]:
            batch = self.tokenizer.pad(
                {"input_ids": [f["input_ids"] for f in features],
                 "attention_mask": [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]},
                padding=True, max_length=self.max_length, return_tensors="pt")
        else:
            batch = self.tokenizer([f["text"] for f in features], padding=True,
                                   truncation=True, max_length=self.max_length, return_tensors="pt")
        batch["labels"] = self._build_labels(batch["input_ids"], batch["attention_mask"])
        return batch


def parse_sql(raw_output: str) -> str:
    marker = "SQL output:"
    idx = raw_output.rfind(marker)
    q = raw_output[idx + len(marker):] if idx != -1 else raw_output
    q = q.strip("`\n ")
    if q.lower().startswith("sql\n"):
        q = q[4:]
    return q.strip("`\n ").strip()


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning", "").strip() and rec.get("sql", "").strip():
                records.append(rec)
    return records


def train(args):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    records = load_records(args.train_data)
    print(f"[{args.variant}] Loaded {len(records)} matched training examples")

    kwargs = {"cache_dir": args.hf_cache} if args.hf_cache else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    instruction = SQL_COT_INSTRUCTION if args.variant == "cot" else SQL_BASELINE_INSTRUCTION

    def make_text(rec):
        user = instruction.format(schema=rec["schema"], question=rec["question"])
        if args.variant == "cot":
            assistant = f"Reasoning:\n{rec['reasoning']}\n\nSQL output: {rec['sql']}"
        else:
            assistant = rec["sql"]
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    texts = [make_text(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Example (first 600 chars):\n{texts[0][:600]}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, attn_implementation="eager", device_map="auto", **kwargs)
    lora = LoraConfig(r=64, lora_alpha=64, lora_dropout=0.05,
                      target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    collator = CompletionOnlyCollator(tokenizer, "<start_of_turn>model\n", max_length=1600)

    sft_kwargs = dict(
        output_dir=args.output_dir, num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4, gradient_accumulation_steps=8,
        learning_rate=2e-5, optim="paged_adamw_8bit", bf16=True,
        logging_steps=10, save_steps=200, save_total_limit=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", dataset_text_field="text")
    try:
        targs = SFTConfig(**sft_kwargs, max_length=1600)
    except TypeError:
        targs = SFTConfig(**sft_kwargs, max_seq_length=1600)

    tk = dict(model=model, args=targs, train_dataset=dataset, peft_config=lora, data_collator=collator)
    try:
        trainer = SFTTrainer(**tk, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**tk, tokenizer=tokenizer)

    resume = None
    if os.path.isdir(args.output_dir):
        ckpts = sorted([d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if ckpts:
            resume = os.path.join(args.output_dir, ckpts[-1])
            print(f"Resuming from {resume}")

    print(f"[{args.variant}] Training ~{len(dataset) // 32} steps...")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume)
    print(f"Training done in {(time.time() - start) / 60:.1f} min.")
    final = os.path.join(args.output_dir, "final")
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    print(f"[{args.variant}] Adapter saved to {final}")


def evaluate(args):
    import evaluate
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"cache_dir": args.hf_cache} if args.hf_cache else {}
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, **kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, attn_implementation="eager", device_map="auto", **kwargs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    instruction = SQL_COT_INSTRUCTION if args.variant == "cot" else SQL_BASELINE_INSTRUCTION
    max_new = 1024 if args.variant == "cot" else 512

    with open(args.test_data) as f:
        examples = [json.loads(line) for line in f]
    print(f"[{args.variant}] Test set: {len(examples)} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_path = os.path.join(args.output_dir, f"predictions_{args.tag}_{args.variant}.jsonl")
    done = 0
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            done = sum(1 for _ in f)
        print(f"Resuming from {done}")

    bs = args.batch_size
    start = time.time()
    with open(pred_path, "a") as out_f:
        for i in range(done, len(examples), bs):
            batch = examples[i:i + bs]
            prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction.format(
                    schema=ex["schema"], question=ex["question"])}],
                add_generation_prompt=True, tokenize=False) for ex in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=max_new,
                                        pad_token_id=tokenizer.eos_token_id)
                new = tokens[:, inputs.input_ids.shape[1]:]
                raws = tokenizer.batch_decode(new, skip_special_tokens=True)
            for ex, raw in zip(batch, raws):
                out_f.write(json.dumps({
                    "instance_id": ex.get("instance_id", ""),
                    "db_id": ex.get("db_id", ""),  # for Spider execution eval
                    "predicted_sql": parse_sql(raw),
                    "reference_sql": ex["sql"],
                    "sql_complexity": ex.get("sql_complexity", "unknown"),
                    "raw_output": raw,
                }) + "\n")
            if (i + bs) % 100 < bs:
                out_f.flush()
                rate = (i + bs - done) / (time.time() - start)
                print(f"  [{i + bs}/{len(examples)}] {rate:.2f}/s")

    with open(pred_path) as f:
        recs = [json.loads(line) for line in f]
    preds = [r["predicted_sql"] for r in recs]
    refs = [r["reference_sql"] for r in recs]
    gleu = evaluate.load("google_bleu").compute(
        predictions=preds, references=[[r] for r in refs])["google_bleu"]
    norm = lambda s: " ".join(s.split())
    em = sum(1 for p, r in zip(preds, refs) if norm(p) == norm(r))

    # per-complexity EM — the theory predicts CoT helps MOST on compositional queries
    by_cx = defaultdict(lambda: [0, 0])
    for r in recs:
        cx = r.get("sql_complexity", "unknown")
        by_cx[cx][1] += 1
        if norm(r["predicted_sql"]) == norm(r["reference_sql"]):
            by_cx[cx][0] += 1
    per_complexity = {cx: {"em": c / n, "n": n} for cx, (c, n) in sorted(by_cx.items())}

    metrics = {"variant": args.variant, "n": len(recs), "gleu": gleu,
               "string_em": em / len(recs), "em_count": em, "per_complexity": per_complexity}
    with open(os.path.join(args.output_dir, f"metrics_{args.tag}_{args.variant}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[{args.variant}] GLEU={gleu:.4f}  String EM={em/len(recs):.4f} ({em}/{len(recs)})")
    for cx, s in per_complexity.items():
        print(f"    {cx:<28} EM={s['em']:.4f} (n={s['n']})")


def main():
    p = argparse.ArgumentParser(description="Matched direct-vs-CoT SQL experiment (positive control)")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--variant", required=True, choices=["direct", "cot"])
    p.add_argument("--train-data")
    p.add_argument("--test-data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--adapter-path")
    p.add_argument("--hf-cache", default=None)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--tag", default="sql",
                   help="Prefix for output files (use 'spider' for the Spider run).")
    args = p.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
