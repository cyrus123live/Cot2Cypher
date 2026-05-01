#!/usr/bin/env python3
"""
QLoRA fine-tuning on ZOGRASCOPE CoT traces.

Mirrors the Neo4j training setup (same QLoRA config) but with:
- ZOGRASCOPE training data (Pole schema, inline-WHERE syntax)
- Same prompt format we used for Neo4j
- Output to ~/scratch/zograscope_adapter/

Usage (from drac_train_zograscope.sh):
    python drac_train_zograscope.py \\
        --train-data /path/to/zograscope_cot_traces.jsonl \\
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
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

BASE_MODEL = "google/gemma-2-9b-it"

# Same prompt format as Neo4j training
COT_INSTRUCTION = (
    "Generate Cypher statement to query a graph database.\n"
    "Use only the provided relationship types and properties in the schema.\n"
    "Schema: {schema}\n"
    "Question: {question}\n\n"
    "Think step by step, then provide the Cypher query."
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune on ZOGRASCOPE CoT traces")
    parser.add_argument("--train-data", required=True, help="Path to ZOGRASCOPE CoT JSONL")
    parser.add_argument("--output-dir", required=True, help="Adapter output directory")
    parser.add_argument("--hf-cache", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    # Load training data
    records = []
    with open(args.train_data) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("reasoning", "").strip():
                records.append(rec)
    print(f"Loaded {len(records)} ZOGRASCOPE training examples")

    # Tokenizer
    kwargs = {"cache_dir": args.hf_cache} if args.hf_cache else {}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, **kwargs)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format examples
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
    print(f"Example (first 600 chars):\n{texts[0][:600]}")

    # QLoRA config (matching Neo4j baseline)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        attn_implementation="eager",
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

    # Completion-only collator
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        optim="paged_adamw_8bit",
        max_seq_length=1600,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
    )

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
    print(f"Steps: ~{len(dataset) // (4 * 8)}")
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
