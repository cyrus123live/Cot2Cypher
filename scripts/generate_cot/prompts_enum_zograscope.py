"""E4: HOLISTIC + EXPLICIT RELATIONSHIP-ENUMERATION reasoning (ZOGRASCOPE).

Targets BOTH CoT sub-mechanisms identified in MECHANISM_ANALYSIS.md:
- fragmentation (fixed by holistic connected-path framing, Test D)
- truncation   (dropped hops, lost in the PLANNING step per diagnostic E1)

The reasoning must FIRST enumerate and COUNT every required relationship in
traversal order, then write the connected query using exactly that many hops.
The enumeration forces the model to commit to the full path up front, countering
the tendency (diagnostics E1-E3) to plan a correct-but-too-short path.
"""

from generate_cot.exemplars_zograscope import EXEMPLARS_ZOG, POLE_SCHEMA_TEXT

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for a Neo4j graph database. \
You will be given a graph schema, a natural language question (with entity values pre-identified), and the correct reference Cypher query. \
Generate a reasoning trace that FIRST enumerates the full relationship chain, then writes the connected query.

Reasoning structure (enumerate-then-construct):
1. **Enumerate the relationship chain**: list EVERY relationship the query needs, in traversal order, and state the count, e.g.:
   "Relationships needed (3): KNOWS_SN -> PARTY_TO -> OCCURRED_AT"
   Include EVERY intermediate hop. Do not stop early; the chain must connect all the nodes mentioned or implied by the question.
2. **State the connected path** using those exact relationships in order, naming each node label, e.g.:
   (Person)-[:KNOWS_SN]-(Person)-[:PARTY_TO]-(Crime)-[:OCCURRED_AT]-(Location)
3. **Anchor filters and return**: which node carries which inline WHERE filter; what is returned/counted.
4. **Write the Cypher** as one connected MATCH using EXACTLY the enumerated relationships (same count, same order). Do not drop or add a hop; do not split into disconnected MATCH clauses.

Notes for this dataset:
- Neo4j 5+ inline WHERE syntax: MATCH (x0:Person WHERE x0.name = "John")-[:KNOWS]-(x1:Person)
- Relationships matched undirected (-[:REL]-). Variables x0, x1, x2, ...

Output exactly:
<reasoning>
Your enumerate-then-construct reasoning here.
</reasoning>
<cypher>
The final Cypher query here.
</cypher>

The query MUST use exactly the relationships you enumerated, in order, fully connected. \
The Cypher in <cypher> must match the reference. Keep reasoning to 3-6 lines. No text outside the tags."""


def build_messages_enum_zograscope(question: str, cypher: str) -> list[dict[str, str]]:
    schema_msg = ("The graph database uses the Pole crime knowledge graph with this schema:\n\n"
                  + POLE_SCHEMA_TEXT)
    messages = [{"role": "system", "content": SYSTEM_MESSAGE + "\n\n" + schema_msg}]
    for ex in EXEMPLARS_ZOG:
        messages.append({"role": "user", "content": _u(ex["question"], ex["cypher"])})
        messages.append({"role": "assistant",
                         "content": _a(_enum_from(ex["cypher"]), ex["cypher"])})
    messages.append({"role": "user", "content": _u(question, cypher)})
    return messages


def _u(question, cypher):
    return f"Question: {question}\n\nReference Cypher: {cypher}"


def _a(reasoning, cypher):
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"


def _enum_from(cypher):
    import re
    rels = re.findall(r'-\[:?(\w+)', cypher)
    path = cypher.split("RETURN")[0].strip()
    chain = " -> ".join(rels) if rels else "(none)"
    return (f"1. Relationships needed ({len(rels)}): {chain}\n"
            f"2. Connected path: {path}\n"
            f"3. Filters are inline; return/count as in the query.\n"
            f"4. Cypher uses exactly those {len(rels)} relationships in one connected MATCH.")
