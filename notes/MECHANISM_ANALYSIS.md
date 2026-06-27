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

## Test D RESULT — holistic-path ablation (causal, 2026-06-25)

Trained a CoT variant whose reasoning states the full connected path explicitly
(no sub-question decomposition), same clean ZOGRASCOPE splits, same pipeline.
Three-way execution accuracy on common instances:

| Split | direct | QDecomp-CoT | holistic-CoT | holistic−QDecomp | holistic−direct |
|-------|:------:|:-----------:|:------------:|:----------------:|:---------------:|
| length (n=1052) | 0.475 | 0.258 | 0.303 | **+0.046** | −0.172 |
| iid (n=768) | 0.826 | 0.609 | 0.710 | **+0.100** | −0.116 |
| compositional (n=1349) | 0.661 | 0.418 | 0.511 | **+0.093** | −0.151 |

**Holistic-CoT beats QDecomp-CoT on every split (+4.6 to +10.0pp), recovering
~40% of the CoT penalty in-distribution.** Decomposition is causally a major
part of the harm. But holistic-CoT still trails direct-answer (−12 to −17pp):
removing decomposition is necessary but not sufficient.

**Mechanism confirmed and localized.** The gain comes specifically from reduced
path fragmentation:

| | QDecomp fragments path | holistic fragments path |
|--|:---:|:---:|
| length | 11% | **2%** (5.5× fewer) |
| regular | 6% | **1%** (6× fewer) |

Holistic reasoning cuts the "split one connected path into disconnected MATCH
clauses" failure ~5–6×. But hop-truncation is unchanged (avg hops length: ref
3.02, QDecomp 2.45, holistic 2.37) — holistic does NOT fix dropped hops.

**Final causal picture (two separable sub-mechanisms):**
1. **Decomposition → fragmentation.** QDecomp's sub-question structure makes the
   model split connected traversals into disconnected MATCH clauses. Causally
   confirmed: holistic reasoning removes the decomposition and cuts fragmentation
   5–6×, recovering ~40% of the penalty.
2. **Residual reasoning cost + truncation.** Even holistic reasoning (a) doesn't
   fix hop-truncation and (b) carries a cost of generating reasoning before the
   answer. This is why holistic-CoT still trails direct-answer.

So: chain-of-thought hurts Text2Cypher because its decompositional bias fragments
connected graph patterns (causally shown) AND reasoning-before-answer carries a
residual cost; direct-answer SFT, which learns the connected pattern holistically
in one shot, avoids both.

## Dropped-hops diagnostics (E1/E2-pre/E3) — 2026-06-25

The half Test D did NOT fix is hop-truncation. Diagnostics on the holistic-CoT
predictions (which still truncate: avg hops length ref 3.02 vs holistic 2.37):

**E1 — where is the hop lost? PLANNING, 99–100%.** Of dropped-hop failures, the
reasoning ITSELF omits the hop (392/394 length, 217/218 regular); the query
faithfully follows the truncated reasoning. The model reasons out a shorter path
— it is not a reasoning→query translation error.

**E2-pre — which hop?** Dropped hops are mid-path (43%) and terminal (48%), rarely
the starting anchor (9%). Type is ~50/50 connector (no filter) vs filtered node —
so it is NOT just "unanchored connectors drop out"; the model shortens paths even
through named/filtered nodes.

**E3 — repair/subpath test.** 93% of dropped-hop failures have a predicted path
that is a clean in-order SUBPATH of the reference — everything else (included hops,
order, filters) is correct; the only error is the missing hop(s). Truncation is
causally THE error, not one symptom among compounding ones.

**Conclusion:** on long paths the model's reasoning commits to a correct-but-too-
short path (right structure/filters, a hop or two missing mid/end). Because it is
a planning failure (E1) and the rest is correct (E3), the fix must force the
reasoning to commit to the full relationship list up front → motivates E4.

## E4 — holistic + explicit relationship enumeration (in progress)

Reasoning that FIRST enumerates and counts every required relationship in order
("Relationships needed (3): KNOWS_SN -> PARTY_TO -> OCCURRED_AT"), keeps the
holistic connected-path framing (which fixed fragmentation in Test D), then
constructs using exactly that many hops. Targets BOTH sub-mechanisms:
fragmentation (holistic) + truncation (enumeration).
- If E4 closes the gap to direct-answer → POSITIVE result: "CoT can match
  direct-answer for Cypher with holistic, hop-enumerated reasoning."
- If E4 fixes truncation but still trails → residual reasoning-before-answer cost.
Format verified: teacher states the relationship count matching reference hops
20/20 in pilot. Variant plumbing: `--enum` (generation), `VARIANT=enum` (train/eval).

## E4 RESULT — enumeration does NOT close the gap (2026-06-27)

Full ablation ladder, execution accuracy (common instances):

| Config | length | iid | compositional |
|--------|:------:|:---:|:-------------:|
| direct-answer | 0.475 | 0.826 | 0.661 |
| QDecomp-CoT | 0.258 | 0.609 | 0.418 |
| **Holistic-CoT** | **0.303** | **0.710** | **0.511** |
| Enum-CoT (E4) | 0.231 | 0.669 | 0.429 |

Truncation diagnostic (length, hop deficit vs reference 3.02 / regular vs 2.12):

| Config | length deficit | regular deficit | fragmentation (length) |
|--------|:---:|:---:|:---:|
| QDecomp | +0.57 | +0.06 | 11% |
| Holistic | +0.64 | +0.10 | 2% |
| Enum(E4) | +0.58 | **+0.01** | 3% |

**Enumeration FAILED to close the gap — it underperforms Holistic everywhere.**
Findings:
1. Enumeration fixes truncation only on SHORT paths (regular deficit +0.06→+0.01,
   best of any arm) but NOT on long paths (length +0.57→+0.58, unchanged). Telling
   the model "you need 3 relationships" does not make it produce 3 correct hops
   when the path is long — the planning-stage shortening (E1) is deeper than a
   counting prompt can fix.
2. The added enumeration scaffolding has a COST: reintroduces fragmentation
   (3% vs Holistic's 2%) and loses accuracy vs Holistic on every split. Listing-
   then-building partially recreates the decompose-then-assemble problem Holistic
   avoids. More reasoning structure ≠ better; minimal holistic framing is best.
3. **No reasoning-format intervention closes the gap to direct-answer.** Holistic
   (minimal connected-path) is the CoT ceiling and still trails by 12–17pp.

## Final conclusion (mechanism, fully bounded)

The CoT penalty for Text2Cypher decomposes into:
- **Fragmentation** — splitting connected paths into disconnected MATCH clauses.
  CAUSED by decomposition; FIXED by holistic framing (recovers +4.6–10pp, cuts
  fragmentation 5–6×). [Test D]
- **Truncation** — planning a correct-but-too-short path on long traversals. A
  planning-stage failure (99–100%, E1); 93% are clean subpaths (E3). RESISTS
  prompt intervention: enumeration fixes it on short paths only, and costs more
  than it saves on long ones. [E4]
- **Residual** — even the best reasoning format trails direct-answer; the deficit
  is intrinsic to generating reasoning before a connected graph pattern.

Direct-answer SFT wins by learning the connected pattern holistically in one shot,
incurring none of the three. **The negative result is not just empirical but
mechanistically explained and bounded: we identify the failure modes, causally fix
the fixable one, and show the other resists the natural intervention.**

## CORRECTION — truncation is NOT the primary cause; filter-value corruption is (2026-06-27)

Prompted by the question "if E4 couldn't verify truncation, is it even the cause?",
a proper decomposition of ALL holistic-CoT failures (n≈1200, execution ground
truth) overturns the truncation framing:

| Failure attribute (non-exclusive) | share |
|-----------------------------------|:-----:|
| truncation ALONE (no co-occurring error) | **2%** |
| has truncation (with other errors) | 31% |
| wrong filter VALUE | (CoT-specific, see below) |
| wrong RETURN | 21% |

**Truncation alone explains only 2%.** E4 corroborated behaviorally: it fixed
truncation on the regular split (hop deficit +0.06→+0.01) yet accuracy still
dropped. The truncation framing is dead.

### The robust CoT-specific culprit: filter-value corruption

Variable-name-independent value-set comparison (robust):

| Split | CoT wrong filter-values | direct-answer | ratio |
|-------|:-----------------------:|:-------------:|:-----:|
| length | 16% | 4% | 3.7× |
| regular | 5% | 1% | 5.0× |

**92% of CoT's wrong values already appear in its reasoning trace** — the value
is corrupted DURING reasoning, then copied into the query. Examples: question
"vehicle-related crimes in BL5" → reference filters only `BL5`, CoT invents a
spurious `"vehicle"` filter; "crimes at homes" → CoT adds `"home"`. ZOGRASCOPE
hands the entity values to the model (e.g. `[Person] = Bonnie`); direct-answer
copies them straight to the query in one pass, CoT re-states them in prose first
and each re-statement risks corruption.

**Honesty bounds (do NOT over-claim):**
- The sub-type is mixed: of value mismatches, ~14% add spurious values, ~21% swap,
  ~65% drop. No single clean sub-mechanism ("over-constraining") dominates.
- Filter-value errors are CoT-specific and 4× worse, but touch only ~16% of length
  predictions — they do NOT explain the whole 73%→52% gap. The remainder is
  diffuse: right relationships, right values, subtly wrong assembly.

### Unifying interpretation (error accumulation on a copy-and-place task)

ZOGRASCOPE is largely a COPYING task — values are handed to the model; the job is
placing them in the right structural slots. Direct-answer does this copy in one
pass. CoT re-states everything (values, paths, structure) in prose reasoning first,
and every re-statement is a chance to corrupt/add/drop/swap — error accumulation in
a generation chain. Filter-value corruption (4×) is the clearest fingerprint; the
diffuse remainder is the same effect spread across structure. This is why MORE
reasoning made it strictly worse (Enum < Holistic < direct) and the MINIMAL
reasoning format (Holistic) was the best CoT variant. Reasoning-before-answer is a
liability, not an asset, for copy-and-place query generation.

## Why this differs from the famous Text-to-SQL CoT results

1. **The SQL evidence is thinner than its headlines.** The cleanest clean
   comparison (STaR-SQL, small model, CoT-SFT vs direct-SFT, same data) is just
   **+6.4pp** (68.6→75.0); the famous "+18pp" was the ORM verifier doing
   best-of-N@16. Most other celebrated results PROMPT frontier models or BUNDLE
   CoT with RL/DPO. So the gap to explain is ~+6pp SQL vs −15pp Cypher, structural.
2. **Decomposition fits SQL, fragments Cypher.** [verified, Test D] SQL is
   compositional (clauses/subqueries/CTEs) — sub-question decomposition maps onto
   it. Cypher's MATCH is a single connected pattern; decomposition fragments it.
3. **Constrained output space.** [verified, self-consistency] SQL has many
   equivalent forms → room for reasoning to explore and for verifiers to select
   (where STaR's real gain came from). Cypher is canonical; string-voting failed,
   only execution-voting helped slightly. The selection mechanism behind SQL's big
   numbers has little to select over in Cypher.
4. **Copy-and-place vs infer-and-compute.** [partially verified] SQL benchmarks
   require inferring literal values (cell matching, units) — reasoning helps.
   ZOGRASCOPE hands values over; reasoning keeps the corruption downside, loses the
   inference upside.
5. **Caveat (about us, not just Cypher):** our direct-answer baseline is unusually
   strong, compressing headroom. CoT helps most when the task is hard relative to
   the base model; a weaker baseline could show CoT "helping" — exactly the
   illusion the confounded baseline created before the matched control.

## Caveats

- String-EM used for Test A binning; B/C use execution ground truth.
- "hops" = count of `-[` relationship markers in the reference; robust but coarse.
- Connector heuristic (Test C) is crude (filter-less adjacent node).
- The STaR-trained models had small training sets; the ZOGRASCOPE clean models are
  the primary evidence here.
