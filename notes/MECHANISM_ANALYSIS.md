# Why CoT Fails for Text2Cypher: Mechanism Analysis

Analysis of *how* chain-of-thought distillation fails relative to direct-answer SFT,
on the clean (leakage-free) ZOGRASCOPE splits with execution ground truth.
Compiled 2026-06-23.

## Setup

- Clean ZOGRASCOPE splits (length + iid + compositional), CoT vs direct-answer,
  same matched pipeline, only the reasoning prefix differs.
- Failure unit: cases where direct-answer is execution-correct and CoT is
  execution-wrong (n=956 across the splits).
- Three falsification tests (A/B/C). Two falsified the initial framing; the
  surviving claim is sharper and strongly supported.

## Test A — does the accuracy gap scale with hop count? ❌ FALSIFIED (as stated)

| ref hops | n | direct EM | CoT EM | gap |
|---------:|--:|:---------:|:------:|:---:|
| 1 | 555 | 0.847 | 0.705 | +0.142 |
| 2 | 864 | 0.588 | 0.257 | **+0.331** |
| 3 | 1627 | 0.344 | 0.143 | +0.201 |
| 4 | 324 | 0.198 | 0.056 | +0.142 |

The absolute gap **peaks at 2 hops and declines** — not monotone in path length.
The naive "deficit grows with hops" claim is wrong. (CoT's *relative* deficit does
grow — 1.2× at 1 hop to 3.5× at 4 — but both models approach the floor at high
hops, so no clean scaling claim is defensible.)

## Test B — what is the structure of CoT's failures? (execution ground truth)

956 cases, direct correct & CoT wrong:

| Failure structure | share |
|-------------------|:-----:|
| Dropped hops (subset of ref relationships) | 32% |
| Added hops (superset) | 5% |
| **Same relationships, wrong connected structure** | **49%** |
| Disjoint / unrelated relationships | **3%** |

The "primarily truncation" framing is **false** (dropping is only 32%). The
decisive number is the last row: **only 3% of CoT failures use unrelated
relationships — 97% use the correct relationships.** Schema-linking (the InterCOL
step) works; CoT reliably identifies *which* relationships the query needs. The
failure is in **composing them into the correct connected pattern**: wrong order,
fragmentation into disconnected MATCH clauses (13% of the same-rel failures),
dropped/added hops, misplaced filters.

## Test C — are dropped hops "connector" nodes? ~ weak

Of dropped relationships, 33% connect filter-less connector nodes — a real
sub-pattern, not the dominant story.

## The surviving, falsification-tested hypothesis

> **CoT's failure for Text2Cypher is compositional, not relational.** The
> decomposition correctly identifies *which* schema relationships to use (InterCOL
> works — 97% of failures use the right relationships), but the
> decompose-then-reassemble process produces the wrong *connected structure*. The
> model has the right pieces and assembles them wrong.

Why direct-answer wins: it learns the connected graph pattern holistically, in one
shot, and never decomposes it. CoT talks itself out of the connected path.

Why this is specific to graph queries (and why SQL CoT results don't transfer):
a Cypher pattern is a single connected object that must be assembled as a whole;
SQL composes naturally from sub-queries/CTEs, so decomposition is the right
inductive bias there and the wrong one here.

## Causal test (D) — set up by this analysis

If the failure is *assembly*, a **holistic-path** reasoning format (write the full
connected path explicitly; no sub-question decomposition) should fix the 49%
wrong-structure failures while preserving the 97% correct relationship
identification.
- Holistic-path CoT closes the gap → decomposition is causally the assembly-breaker.
- Holistic-path CoT still fails → the cause is generating-reasoning-before-answer,
  not decomposition specifically.
Either outcome sharpens the claim. Implemented as the `prompts_holistic` reasoning
format + STaR-style regeneration/retrain.

## Caveats

- String-EM used for Test A binning; B/C use execution ground truth.
- "hops" = count of `-[` relationship markers in the reference; robust but coarse.
- Connector heuristic (Test C) is crude (filter-less adjacent node).
- The STaR-trained models had small training sets; the ZOGRASCOPE clean models are
  the primary evidence here.
