"""HOLISTIC-PATH reasoning for ZOGRASCOPE (Test D ablation, Pole schema).

Same contrast as prompts_holistic.py but for the Pole graph / inline-WHERE
dialect. Reasoning states the full connected path explicitly (no sub-question
decomposition), to test whether removing decomposition fixes the
mis-assembly failure mode identified in notes/MECHANISM_ANALYSIS.md.
"""

from generate_cot.exemplars_zograscope import EXEMPLARS_ZOG, POLE_SCHEMA_TEXT

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for a Neo4j graph database. \
You will be given a graph schema, a natural language question (with entity values pre-identified), and the correct reference Cypher query. \
Generate a reasoning trace that explains the query as a single CONNECTED GRAPH PATTERN — NOT by decomposing the question into sub-questions.

Holistic-path reasoning structure:
1. **State the full connected path** as one object in traversal order, naming every node label and relationship type, including EVERY intermediate (connector) node. Do not collapse or skip hops.
2. **Anchor filters and return**: which node carries which inline WHERE filter; what is returned/counted.
3. **Write the Cypher** as one connected MATCH that preserves the entire path. Do NOT split a single connected path into separate disconnected MATCH clauses.

Notes for this dataset:
- Neo4j 5+ inline WHERE syntax: MATCH (x0:Person WHERE x0.name = "John")-[:KNOWS]-(x1:Person)
- Relationships matched undirected (-[:REL]-). Variables named x0, x1, x2, ...

Output exactly:
<reasoning>
Your holistic-path reasoning here.
</reasoning>
<cypher>
The final Cypher query here.
</cypher>

The path must stay CONNECTED end-to-end; every relationship in the reference must appear in order. \
The Cypher in <cypher> must match the reference. Keep reasoning to 3-6 lines. No text outside the tags."""


def build_messages_holistic_zograscope(question: str, cypher: str) -> list[dict[str, str]]:
    schema_msg = ("The graph database uses the Pole crime knowledge graph with this schema:\n\n"
                  + POLE_SCHEMA_TEXT)
    messages = [{"role": "system", "content": SYSTEM_MESSAGE + "\n\n" + schema_msg}]
    for ex in EXEMPLARS_ZOG:
        messages.append({"role": "user", "content": _u(ex["question"], ex["cypher"])})
        messages.append({"role": "assistant",
                         "content": _a(_holistic_from(ex["cypher"]), ex["cypher"])})
    messages.append({"role": "user", "content": _u(question, cypher)})
    return messages


def _u(question, cypher):
    return f"Question: {question}\n\nReference Cypher: {cypher}"


def _a(reasoning, cypher):
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"


def _holistic_from(cypher):
    path = cypher.split("RETURN")[0].strip()
    return (f"1. Connected path: {path}\n"
            f"2. Filters are inline in the path above; return/count as in the query.\n"
            f"3. Cypher keeps the whole connected path in a single MATCH.")
