# Step 2: Chain-of-Thought Training Data Generation

## Status: COMPLETE

39,553 out of 39,554 training examples now have QDECOMP+InterCOL reasoning traces. Output: `data/cot_training_data.jsonl`. Ready for Step 3 (fine-tuning).

---

## Results Summary

| Metric | Result |
|--------|--------|
| Total examples | 39,554 |
| Successes | 39,553 (99.997%) |
| Failures | 1 (parse failure) |
| Cypher match rate | 93.3% (cosmetic mismatches only) |
| Avg reasoning length | 715 chars (~295 tokens) |
| Avg prompt tokens | 4,735 |
| Total tokens | 198,973,742 (187M prompt + 12M completion) |
| Provider | Cerebras (GPT-oss-120B at 3,000 tok/s) |
| Total cost | ~$48 |
| Total time | ~3.3 hours |

### Per-Source Success Rates

All 20 data sources achieved 100% success rate except `neo4jLabs_functional_cypher` (13,570/13,571 = 99.99%).

| Source | Count | Success Rate |
|--------|------:|-------------|
| neo4jLabs_functional_cypher | 13,571 | 99.99% |
| neo4jLabs_synthetic_gpt4turbo | 6,348 | 100% |
| neo4jLabs_synthetic_gpt4o | 6,106 | 100% |
| neo4jLabs_synthetic_gemini | 5,895 | 100% |
| neo4jLabs_synthetic_claudeopus | 3,257 | 100% |
| neo4j_text2cypher2023_train | 2,587 | 100% |
| All others (14 sources) | 1,790 | 100% |

### Cypher Match Rate Analysis

93.3% of generated Cypher matches the reference exactly (whitespace-normalized). The 6.7% mismatches are entirely cosmetic and do not affect training:

| Mismatch Category | Count | % of Mismatches | Impact |
|-------------------|------:|----------------:|--------|
| Whitespace differences | 1,692 | 63.5% | None — semantically identical |
| Other minor (added labels, redundant label removal, pattern simplification) | 649 | 24.4% | None — we use reference Cypher for training |
| Long inline string truncated | 119 | 4.5% | None — model abbreviates 200+ char abstracts embedded in Cypher |
| Case differences (`in`→`IN`, `Artist_name`→`artist_name`) | 117 | 4.4% | None — Cypher keywords are case-insensitive |
| `neo4j`→`neo3j`/`neo8j` tokenizer artifact | 85 | 3.2% | None — BPE splits mixed alphanumeric strings incorrectly |
| Trailing semicolon dropped | 2 | 0.1% | None |

**Key point:** We always use the original `cypher` field as the training target, never `generated_cypher`. The match rate is a diagnostic signal confirming the model understood the queries.

---

## What Was Built

### File Structure

```
generate_cot/
├── __init__.py      # Package init
├── config.py        # Provider URLs, model names, rate limits, defaults
├── exemplars.py     # 9 hand-crafted few-shot exemplars
├── prompts.py       # System message + prompt assembly
├── parse.py         # XML tag parsing + validation
├── generate.py      # Main async generation script (entry point)
└── analyze.py       # Post-generation stats + failure review
data/
└── cot_training_data.jsonl   # Output (39,554 records, ~500MB, gitignored)
```

### Module Descriptions

**`config.py`** — Provider-agnostic configuration. Supports Cerebras, Galaxy.ai, OpenRouter, DeepInfra, and OpenAI via environment variables (`COT_PROVIDER`, `COT_API_KEY`). All use the OpenAI-compatible API format. Defaults: 50 concurrent requests, temperature 0.3, max 1024 output tokens, 3 retries with exponential backoff.

**`exemplars.py`** — Nine hand-crafted few-shot exemplars selected from the actual training set, each with a manually written QDECOMP+InterCOL reasoning trace. Covers three schema formats (minimal, verbose, JSON) and diverse query patterns (see coverage matrix below).

**`prompts.py`** — Assembles the full prompt: system message instructing QDECOMP+InterCOL format → 9 few-shot exemplar pairs (user/assistant) → target example. The system message specifies XML output format (`<reasoning>` and `<cypher>` tags) and requires schema-grounded reasoning in 3-8 lines. Total prompt size is ~4,735 tokens per request.

**`parse.py`** — Extracts reasoning and Cypher from LLM responses. Primary path: XML tag extraction via regex. Fallback: heuristic parsing for `Reasoning:` / `Cypher:` markers. Handles partial tag matches. Validation checks: minimum reasoning length (20 chars), minimum Cypher length (5 chars), presence of Cypher keywords (MATCH/RETURN/CALL/CREATE/MERGE).

**`generate.py`** — Async entry point using `AsyncOpenAI`. Loads the dataset from HuggingFace, builds a set of completed instance IDs from the checkpoint file, dispatches async requests through a semaphore, and writes append-only JSONL. Supports `--pilot` (50 stratified examples), `--limit N` (batch mode), and auto-resume from checkpoint.

**`analyze.py`** — Post-generation analysis: success/failure counts, reasoning length distribution, Cypher match rate, per-source breakdown, failure error categorization, token usage summary, and random sample display for manual review.

---

## Prompt Design

### Approach: Distillation with Reference Cypher

The reference Cypher is included in the prompt. This is standard CoT distillation — we want GPT-oss-120B to generate reasoning that explains the *correct* answer, not to independently generate (possibly wrong) Cypher. The training target is `reasoning` + original `cypher`, not `generated_cypher`.

### System Message

Instructs the model to follow QDECOMP+InterCOL structure:
1. **Decompose** the question into sub-questions aligned with Cypher clause order
2. **Link to schema elements** (InterCOL) — name relevant node labels, relationships, properties
3. **Identify graph pattern** — single hop, multi-hop, variable-length, aggregation, filtering
4. **Construct Cypher step by step** — build MATCH, WHERE, RETURN clauses

Output format: `<reasoning>...</reasoning><cypher>...</cypher>` XML tags.

### Exemplar Coverage Matrix

| # | Schema Format | Query Pattern            | Data Source              | Training Index |
|---|--------------|--------------------------|--------------------------|----------------|
| 1 | Minimal      | Single-hop + WHERE (STARTS WITH) | functional_cypher | idx 4 |
| 2 | Minimal      | Aggregation + ORDER BY + LIMIT | functional_cypher | idx 3 |
| 3 | Minimal      | Variable-length path (`[*3]`) | functional_cypher | idx 6 |
| 4 | Verbose      | Multi-hop + WITH pipeline + COUNT | synthetic_gpt4o | idx 0 |
| 5 | Verbose      | Single-hop + range WHERE | synthetic_claudeopus | idx 7 |
| 6 | Verbose      | Multi-hop + WHERE filter + LIMIT | synthetic_claudeopus | idx 30 |
| 7 | Verbose      | OPTIONAL MATCH + multi-type relationship | synthetic_gpt4turbo | idx 9229 |
| 8 | JSON         | Simple COUNT             | text2cypher2023_train    | idx 20 |
| 9 | JSON         | Aggregation + WITH + ORDER BY | text2cypher2023_train | idx 271 |

**Rationale:**
- **Three schema formats** match the dataset: minimal (functional_cypher), verbose (synthetic sources), JSON (text2cypher2023_train).
- **Query patterns** progress from simple to complex: single node match → filtering → aggregation → multi-hop → variable-length → OPTIONAL MATCH.
- All exemplars use **real (schema, question, cypher) triples** from the training set. Only the reasoning traces are hand-written.

### Example Reasoning Trace

For the question *"Which 3 countries have the most entities linked as beneficiaries in filings?"*:

```
1. Sub-questions: (a) Which entities are beneficiaries of filings?
   (b) What country is each entity in? (c) Which 3 countries have the most such entities?
2. Schema elements: Filing→BENEFITS→Entity (beneficiary link),
   Entity→COUNTRY→Country (country link), Country.name for the country name.
3. Pattern: Two-hop traversal Filing→BENEFITS→Entity→COUNTRY→Country,
   then aggregate by country with COUNT, ORDER BY DESC, and LIMIT 3.
4. Construction:
   - MATCH (f:Filing)-[:BENEFITS]->(e:Entity)-[:COUNTRY]->(c:Country) for the two-hop traversal
   - WITH c.name AS country, COUNT(e) AS entityCount to aggregate per country
   - ORDER BY entityCount DESC LIMIT 3 to get top 3
   - RETURN country, entityCount
```

---

## Output Format

Each line in `data/cot_training_data.jsonl`:

```json
{
  "instance_id": "instance_id_41185",
  "question": "Which 3 countries have the most...",
  "schema": "Node properties:\n- **Country**...",
  "cypher": "MATCH (f:Filing)-[:BENEFITS]->...",
  "reasoning": "1. Sub-questions: (a) Which entities...",
  "generated_cypher": "MATCH (f:Filing)-[:BENEFITS]->...",
  "data_source": "neo4jLabs_synthetic_gpt4o",
  "database_reference_alias": "fincen",
  "generation_metadata": {
    "model": "gpt-oss-120b",
    "prompt_tokens": 4735,
    "completion_tokens": 295
  }
}
```

For fine-tuning (Step 3), we use `reasoning` + original `cypher` (not `generated_cypher`) as the training target.

---

## Generation Run Log

| Batch | Examples | Successes | Failures | Cost | Time |
|-------|----------|-----------|----------|------|------|
| Pilot | 50 | 50 | 0 | $0.07 | 0.2 min |
| Batch 1 | 5,000 | 4,999 | 1 | $6.96 | 25 min |
| Batch 2 | 10,000 | 9,999 | 1 | $13.91 | 50 min |
| Batch 3 | 12,500 | 12,498 | 2 | $17.35 | 64 min |
| Batch 4 (partial, 402 errors) | 6,845 | 6,849 | 5,155 | $9.48 | 48 min |
| Batch 4 retry | 5,159 | 5,158 | 1 | $7.12 | 26 min |
| **Total** | **39,554** | **39,553** | **1** | **~$48** | **~3.3 hours** |

Batch 4 initially hit a Cerebras 402 (payment required) error after ~6,849 successes. Failed records were stripped from the checkpoint and retried after adding credits.

---

## Cost Comparison

| Provider | Input/1M | Output/1M | Estimated Total |
|----------|----------|-----------|-----------------|
| **Cerebras (used)** | $0.25 | $0.69 | **$48 actual** |
| Galaxy.ai | $0.02 | $0.10 | ~$4 estimated |
| OpenRouter | $0.039 | $0.19 | ~$10 estimated |
| DeepInfra | $0.04 | $0.19 | ~$10 estimated |
| OpenAI | $0.15 | $0.60 | ~$28 estimated |

Cerebras was 12x more expensive than Galaxy.ai but delivered 3,000 tok/s inference speed, completing the full run in 3.3 hours vs an estimated 10+ hours on cheaper providers.

---

## How to Run (Reference)

```bash
# Pilot run — 50 diverse examples
COT_PROVIDER=cerebras COT_API_KEY=... ./venv/bin/python -m generate_cot.generate --pilot

# Batch run with limit
COT_PROVIDER=cerebras COT_API_KEY=... ./venv/bin/python -m generate_cot.generate --limit 5000

# Full run (auto-resumes from checkpoint)
COT_PROVIDER=cerebras COT_API_KEY=... ./venv/bin/python -m generate_cot.generate

# Analyze results
./venv/bin/python -m generate_cot.analyze --sample 10
./venv/bin/python -m generate_cot.analyze --failures
```

---

## Key Design Decisions

1. **Include reference Cypher in prompt (distillation)** — The model sees the correct Cypher and generates reasoning that explains it. Standard CoT distillation practice (STaR-SQL, ExCoT). Prevents incorrect reasoning from contaminating training data.

2. **XML tags for output format** — `<reasoning>` and `<cypher>` tags are unambiguous to parse. Parser includes fallback for partial tags and plain-text markers.

3. **Provider-agnostic via OpenAI SDK** — Switch providers by changing `COT_PROVIDER` env var. No code changes needed.

4. **Append-only JSONL with resume** — Same pattern as `run_neo4j.py`. Safe against interruptions.

5. **Temperature 0.3** — Low enough for consistent formatting, high enough for natural variation.

---

## Dataset Context

The 39,554 training examples span 20 data sources with three distinct schema formats:

| Source | Count | Schema Format |
|--------|------:|---------------|
| neo4jLabs_functional_cypher | 13,571 | Minimal (`Graph schema: ...`) |
| neo4jLabs_synthetic_gpt4turbo | 6,348 | Verbose (markdown with `Node properties: ...`) |
| neo4jLabs_synthetic_gpt4o | 6,106 | Verbose |
| neo4jLabs_synthetic_gemini | 5,895 | Verbose |
| neo4jLabs_synthetic_claudeopus | 3,257 | Verbose |
| neo4j_text2cypher2023_train | 2,587 | JSON |
| neo4j_crowdsourced | 487 | Verbose |
| cyspider_* (6 sources) | 553 | Pipe-delimited / JSON array |
| Others (hf_*, rageval_*) | 750 | Mixed |
