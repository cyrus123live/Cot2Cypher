#!/usr/bin/env python3
"""DRAC pre-flight test: catch TRL/transformers API issues without submitting a SLURM job.

Run on the Fir LOGIN NODE (no GPU needed) after activating the same env you
build in drac_train_zograscope.sh:

    module load python/3.11 scipy-stack gcc arrow
    virtualenv --no-download --system-site-packages /tmp/env
    source /tmp/env/bin/activate
    pip install --no-index torch torchvision
    pip install --no-index transformers accelerate datasets evaluate safetensors sentencepiece protobuf
    pip install --no-index --find-links $HOME/wheels peft trl bitsandbytes
    python ~/scratch/thesis/scripts/smoke_test_drac.py

Tests are layered: each stage runs only if the previous succeeded. Failures
print the offending API and what the script needs.
"""

import json
import os
import sys
import traceback
from pathlib import Path


def stage(n, name):
    print(f"\n=== Stage {n}: {name} ===", flush=True)


def fail(msg):
    print(f"  FAIL: {msg}", flush=True)
    sys.exit(1)


def ok(msg):
    print(f"  OK: {msg}", flush=True)


# Stage 1: Imports
stage(1, "imports")
try:
    import torch
    import datasets
    import transformers
    import peft
    import trl
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  datasets     {datasets.__version__}")
    print(f"  peft         {peft.__version__}")
    print(f"  trl          {trl.__version__}")
except Exception as e:
    fail(f"import error: {e}")
ok("all libraries imported")


# Stage 2: API surface — construct everything that doesn't need a GPU
stage(2, "API surface (SFTConfig, LoraConfig, collator)")

from peft import LoraConfig
from trl import SFTConfig

# 2a. SFTConfig with our exact kwargs (the bit that bit us last time)
sft_kwargs = dict(
    output_dir="/tmp/smoke_out",
    num_train_epochs=1,
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
    cfg = SFTConfig(**sft_kwargs, max_length=1600)
    ok("SFTConfig accepts max_length=1600 (TRL 1.x)")
except TypeError:
    try:
        cfg = SFTConfig(**sft_kwargs, max_seq_length=1600)
        ok("SFTConfig accepts max_seq_length=1600 (TRL 0.x)")
    except TypeError as e:
        fail(f"SFTConfig rejects both max_length and max_seq_length: {e}")
except Exception as e:
    fail(f"SFTConfig blew up: {e}")

# 2b. LoraConfig
try:
    lora = LoraConfig(
        r=64, lora_alpha=64, lora_dropout=0.05,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    ok("LoraConfig built")
except Exception as e:
    fail(f"LoraConfig rejected: {e}")

# 2c. Collator + a tiny tokenizer (cached locally) and one real record
try:
    from transformers import AutoTokenizer
except Exception as e:
    fail(f"AutoTokenizer import: {e}")

# Use any small public tokenizer that's likely cached. Fall back to a string-only
# collator test if no tokenizer is available.
TOKENIZER_CANDIDATES = [
    "google/gemma-2-9b-it",  # what production uses; only loads tokenizer (small)
    "gpt2",  # always cached
]
tokenizer = None
for name in TOKENIZER_CANDIDATES:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            cache_dir=os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE"),
        )
        ok(f"tokenizer loaded: {name}")
        break
    except Exception:
        continue
if tokenizer is None:
    fail("could not load any tokenizer — check HF cache / network")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Reuse the script's collator definition
sys.path.insert(0, str(Path(__file__).parent))
try:
    from drac_train_zograscope import CompletionOnlyCollator
except Exception as e:
    fail(f"importing CompletionOnlyCollator: {e}\n{traceback.format_exc()}")

response_template = "<start_of_turn>model\n"
collator = CompletionOnlyCollator(tokenizer=tokenizer, response_template=response_template, max_length=1600)
ok("CompletionOnlyCollator built")

# Build a fake batch of 2 features
fake_features = [
    {"text": "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello world<end_of_turn>"},
    {"text": "<start_of_turn>user\nFoo<end_of_turn>\n<start_of_turn>model\nBar baz<end_of_turn>"},
]
try:
    batch = collator(fake_features)
    assert "input_ids" in batch and "labels" in batch and "attention_mask" in batch
    # At least some tokens should be unmasked (label != -100)
    unmasked = (batch["labels"] != -100).any().item()
    assert unmasked, "all labels are -100 — collator masked everything"
    ok(f"collator returns batch with shapes {tuple(batch['input_ids'].shape)} and unmasked labels")
except Exception as e:
    fail(f"collator failed on fake batch: {e}\n{traceback.format_exc()}")


# Stage 3: real data round-trip
stage(3, "real data round-trip (load 10 records, format, tokenize)")
data_path = Path(os.environ.get("ZOG_DATA", "/home/cyrusp/scratch/zograscope_cot_traces.jsonl"))
if not data_path.exists():
    # Try local repo
    alt = Path(__file__).parent.parent / "data/zograscope/zograscope_cot_traces.jsonl"
    if alt.exists():
        data_path = alt
if not data_path.exists():
    print(f"  SKIP: no data file at {data_path}")
else:
    records = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            r = json.loads(line)
            if r.get("reasoning", "").strip():
                records.append(r)
    ok(f"loaded {len(records)} records")

    from drac_train_zograscope import COT_INSTRUCTION

    def make_text(record):
        user_content = COT_INSTRUCTION.format(schema=record["schema"], question=record["question"])
        assistant_content = f"Reasoning:\n{record['reasoning']}\n\nCypher output: {record['cypher']}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    try:
        texts = [make_text(r) for r in records]
        ok(f"chat template applied; first text starts: {texts[0][:80]!r}")
    except Exception as e:
        fail(f"apply_chat_template: {e}")

    try:
        from datasets import Dataset
        ds = Dataset.from_dict({"text": texts})
        batch = collator([ds[i] for i in range(min(2, len(ds)))])
        ok(f"real-data batch shape {tuple(batch['input_ids'].shape)}, "
           f"unmasked tokens in batch: {(batch['labels'] != -100).sum().item()}")
    except Exception as e:
        fail(f"dataset+collator on real records: {e}\n{traceback.format_exc()}")


# Stage 4 (optional): SFTTrainer construction with a tiny model on CPU.
# Skip by default since it requires loading model weights and may take a minute.
# Enable with SMOKE_TRAINER=1.
if os.environ.get("SMOKE_TRAINER") == "1":
    stage(4, "SFTTrainer construction (tiny CPU model)")
    try:
        from transformers import AutoModelForCausalLM
        from trl import SFTTrainer
    except Exception as e:
        fail(f"import: {e}")

    try:
        # Load tiny model on CPU (no quantization; just verifying API)
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        ok("tiny gpt2 loaded on CPU")
    except Exception as e:
        fail(f"tiny model load: {e}")

    try:
        # Trainer kwargs — same try/except shape as the real script
        from datasets import Dataset
        tiny_ds = Dataset.from_dict({"text": [
            "<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>",
            "<start_of_turn>user\nFoo<end_of_turn>\n<start_of_turn>model\nBar<end_of_turn>",
        ]})
        # Build a fresh SFTConfig with very small steps
        try:
            small_cfg = SFTConfig(
                output_dir="/tmp/smoke_out",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                logging_steps=1,
                report_to="none",
                dataset_text_field="text",
                max_steps=1,
                bf16=False,
                fp16=False,
                max_length=128,
            )
        except TypeError:
            small_cfg = SFTConfig(
                output_dir="/tmp/smoke_out",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                logging_steps=1,
                report_to="none",
                dataset_text_field="text",
                max_steps=1,
                bf16=False,
                fp16=False,
                max_seq_length=128,
            )

        trainer_kwargs = dict(
            model=model,
            args=small_cfg,
            train_dataset=tiny_ds,
            peft_config=lora,
            data_collator=collator,
        )
        try:
            trainer = SFTTrainer(**trainer_kwargs, processing_class=tokenizer)
            ok("SFTTrainer accepted processing_class= (TRL 1.x)")
        except TypeError:
            trainer = SFTTrainer(**trainer_kwargs, tokenizer=tokenizer)
            ok("SFTTrainer accepted tokenizer= (TRL 0.x)")
    except Exception as e:
        fail(f"SFTTrainer construction: {e}\n{traceback.format_exc()}")

    try:
        trainer.train()
        ok("training step completed")
    except Exception as e:
        fail(f"trainer.train(): {e}\n{traceback.format_exc()}")

print("\nAll stages passed. Safe to sbatch.", flush=True)
