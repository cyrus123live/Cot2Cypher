#!/usr/bin/env python3
"""QLoRA fine-tuning of Llama-3.1-8B-Instruct on Neo4j CoT traces.

Identical procedure to scripts/train_cot.py (Gemma) and
drac_train_zograscope.py (Gemma on ZOGRASCOPE), but with:
- Base model: meta-llama/Llama-3.1-8B-Instruct
- Llama chat template (handled automatically by tokenizer.apply_chat_template)
- Response template for completion-only masking: Llama-3 header pattern
- SDPA attention (Llama-3 doesn't need eager like Gemma-2 does)

Purpose: address Alex's #1 critical gap — every mechanistic claim currently
rests on Gemma-2-9B. Replicating on Llama-3.1-8B answers "is this a Gemma
artifact?" with a clean second data point.

Usage (from drac_train_llama_cot.sh):
    python drac_train_llama_cot.py \\
        --train-data /path/to/cot_training_data.jsonl \\
        --output-dir /path/to/output \\
        --hf-cache /path/to/hf_cache
"""

import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


class CompletionOnlyCollator:
    """Mask loss on everything before the assistant response template.

    Replacement for trl.DataCollatorForCompletionOnlyLM (removed in trl>=0.13).

    Handles both:
      - text features: {"text": "..."} — tokenize, pad, mask.
      - pre-tokenized features: {"input_ids": [...], "attention_mask": [...]} —
        what newer TRL hands us after its own pre-tokenization. Pad, mask.
    """

    def __init__(self, tokenizer, response_template: str, max_length: int):
        self.tokenizer = tokenizer
        self.response_template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        self.max_length = max_length

    def _build_labels(self, input_ids, attention_mask):
        labels = input_ids.clone()
        rt = self.response_template_ids
        rt_len = len(rt)
        for i, ids in enumerate(input_ids):
            ids_list = ids.tolist()
            response_start = None
            for j in range(len(ids_list) - rt_len + 1):
                if ids_list[j : j + rt_len] == rt:
                    response_start = j + rt_len
                    break
            if response_start is None:
                labels[i, :] = -100  # no template found → drop example from loss
            else:
                labels[i, :response_start] = -100
        labels[attention_mask == 0] = -100
        return labels

    def __call__(self, features):
        if features and "input_ids" in features[0]:
            batch = self.tokenizer.pad(
                {
                    "input_ids": [f["input_ids"] for f in features],
                    "attention_mask": [
                        f.get("attention_mask", [1] * len(f["input_ids"]))
                        for f in features
                    ],
                },
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        batch["labels"] = self._build_labels(batch["input_ids"], batch["attention_mask"])
        return batch


BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Identical prompt format to the Gemma Neo4j CoT training
COT_INSTRUCTION = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n\n"
    "Think step by step, then provide the Cypher query."
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1-8B on Neo4j CoT traces")
    parser.add_argument("--train-data", required=True, help="Path to Neo4j CoT JSONL")
    parser.add_argument("--output-dir", required=True, help="Adapter output directory")
    parser.add_argument("--hf-cache", default=None, help="HuggingFace cache directory")
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs (default 1)")
    args = parser.parse_args()

    # Load training data
    records = []
    with open(args.train_data) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning", "").strip():
                records.append(rec)
    print(f"Loaded {len(records)} CoT training examples")

    # Tokenizer
    kwargs = {"cache_dir": args.hf_cache} if args.hf_cache else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format examples using Llama's chat template
    def make_text(record):
        user_content = COT_INSTRUCTION.format(
            schema=record["schema"], question=record["question"]
        )
        assistant_content = (
            f"Reasoning:\n{record['reasoning']}\n\n"
            f"Cypher output: {record['cypher']}"
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    texts = [make_text(r) for r in records]
    dataset = Dataset.from_dict({"text": texts})
    print(f"Dataset: {len(dataset)} examples.")
    print(f"Example (first 700 chars):\n{texts[0][:700]}")

    # QLoRA config (matching Neo4j baseline exactly — only model differs)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        attn_implementation="sdpa",  # Llama-3 is fine with SDPA; eager only needed for Gemma-2
        device_map="auto",
        **kwargs,
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Llama-3 assistant response template for completion-only masking
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = CompletionOnlyCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        max_length=1600,
    )

    # Training config
    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
    )
    try:
        training_args = SFTConfig(**sft_kwargs, max_length=1600)
    except TypeError:
        training_args = SFTConfig(**sft_kwargs, max_seq_length=1600)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        data_collator=collator,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)

    # Resume from latest checkpoint if any
    resume_from = None
    if os.path.isdir(args.output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoints:
            resume_from = os.path.join(args.output_dir, checkpoints[-1])
            print(f"Resuming from {resume_from}")

    print("Starting training...")
    print(f"Steps: ~{len(dataset) // (4 * 8) * args.num_epochs}")
    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.1f} minutes.")

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
