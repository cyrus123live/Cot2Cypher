"""HOLISTIC-PATH reasoning prompts — the Test D ablation.

Contrast with prompts.py (QDecomp+InterCOL, which decomposes the question into
sub-questions). The mechanism analysis found CoT identifies the right
relationships (97%) but mis-assembles the connected pattern. This format
removes the sub-question decomposition and instead writes the FULL CONNECTED
PATH explicitly as one object, then the query — testing whether decomposition
is causally the assembly-breaker.

The teacher is GIVEN the reference Cypher (distillation, like prompts.py), but
the reasoning it must produce is holistic-path, not decomposed.
"""

from generate_cot.exemplars import EXEMPLARS

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for Neo4j graph databases. \
You will be given a graph schema, a natural language question, and the correct reference Cypher query. \
Your task is to generate a reasoning trace that explains the query as a single CONNECTED GRAPH PATTERN — \
NOT by decomposing the question into sub-questions.

Use this holistic-path reasoning structure:
1. **State the full connected path** as one object, naming every node label and relationship type in traversal order, e.g.:
   (Person)-[:KNOWS]-(Person)-[:PARTY_TO]-(Crime)-[:OCCURRED_AT]-(Location)
   Include EVERY intermediate node — do not collapse or skip connector nodes.
2. **Anchor the filters and the return**: which node carries which property filter, and what is returned/aggregated.
3. **Write the Cypher** as a single connected MATCH that preserves the whole path (avoid splitting one path into separate disconnected MATCH clauses).

Output your answer in this exact format:
<reasoning>
Your holistic-path reasoning here.
</reasoning>
<cypher>
The final Cypher query here.
</cypher>

Important:
- Reference specific node labels, relationship types, and properties FROM THE SCHEMA.
- Keep the path CONNECTED end-to-end; every relationship in the reference must appear, in order.
- The final Cypher in <cypher> tags must match the reference query.
- Keep reasoning concise (3-6 lines). No text outside the XML tags."""


def build_messages_holistic(schema: str, question: str, cypher: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for ex in EXEMPLARS:
        messages.append({"role": "user",
                         "content": _u(ex["schema"], ex["question"], ex["cypher"])})
        messages.append({"role": "assistant",
                         "content": _a(_holistic_reasoning_from(ex), ex["cypher"])})
    messages.append({"role": "user", "content": _u(schema, question, cypher)})
    return messages


def _u(schema, question, cypher):
    return f"Schema:\n{schema}\n\nQuestion: {question}\n\nReference Cypher: {cypher}"


def _a(reasoning, cypher):
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"


def _holistic_reasoning_from(ex):
    """Build a holistic-path exemplar reasoning from an existing QDecomp exemplar.

    We don't have hand-written holistic exemplars, so we synthesize a short
    path-first reasoning from the exemplar's reference Cypher. This only seeds
    the FEW-SHOT format; the teacher generates fresh holistic reasoning for each
    target. Kept deliberately simple — the format matters, not the exact prose.
    """
    import re
    cy = ex["cypher"]
    # crude path string: node labels and rels in order
    toks = re.findall(r'\(([^)]*)\)|-\[:?(\w+)[^\]]*\]-?(>|<)?', cy)
    return ("1. Connected path: " + cy.split("RETURN")[0].strip() + "\n"
            "2. Filters/return as in the query.\n"
            "3. Cypher preserves the full connected path in a single MATCH.")
