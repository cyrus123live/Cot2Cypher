"""ZOGRASCOPE-specific system message and prompt assembly.

Differences from the Neo4j prompts:
- Single fixed Pole schema is used (not embedded per-example)
- All Cypher uses Neo4j 5+ inline WHERE syntax
- Examples use ZOGRASCOPE entity-linking format ([Person] = X) in the question
"""

from generate_cot.exemplars_zograscope import EXEMPLARS_ZOG, POLE_SCHEMA_TEXT

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for a Neo4j graph database. \
You will be given a graph schema, a natural language question (with entity values pre-identified), and the correct reference Cypher query. \
Your task is to generate a step-by-step reasoning trace that explains how to arrive at the given Cypher query from the question and schema.

Follow the QDECOMP+InterCOL reasoning structure:
1. **Decompose the question** into sub-questions aligned with Cypher clause order (MATCH → WHERE → RETURN/aggregation).
2. **Link to schema elements** (InterCOL): For each sub-question, name the relevant node labels, relationship types, and properties from the schema.
3. **Identify the graph pattern**: single hop, multi-hop traversal, undirected match, count/aggregation, top-k via ORDER BY + LIMIT, etc.
4. **Construct the Cypher step by step**: Build MATCH patterns with inline WHERE filters, add additional MATCH clauses if needed, construct RETURN with aggregations or ORDER BY/LIMIT.

Important notes for this dataset:
- This dataset uses Neo4j 5+ inline WHERE syntax: e.g., MATCH (x:Person WHERE x.name = "John")-[:KNOWS]-(y:Person)
- Relationships are matched undirected (using -[:REL]-) by convention, even when the schema implies a direction.
- Variables are systematically named x0, x1, x2, ...

Output your answer in this exact format:
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<cypher>
The final Cypher query here.
</cypher>

The reasoning must reference specific node labels, relationship types, and properties FROM THE SCHEMA. \
The Cypher in <cypher> tags must match the reference query exactly. \
Keep reasoning concise but complete (typically 4-8 lines). \
Do not include any text outside the XML tags."""


def build_messages_zograscope(question: str, cypher: str) -> list[dict[str, str]]:
    """Build messages for ZOGRASCOPE generation.

    The Pole schema is shared across all examples and provided once
    in the system message context.
    """
    schema_message = (
        "The graph database uses the Pole crime knowledge graph with this schema:\n\n"
        + POLE_SCHEMA_TEXT
    )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE + "\n\n" + schema_message}
    ]

    for ex in EXEMPLARS_ZOG:
        messages.append({
            "role": "user",
            "content": _format_user_message(ex["question"], ex["cypher"]),
        })
        messages.append({
            "role": "assistant",
            "content": _format_assistant_message(ex["reasoning"], ex["cypher"]),
        })

    messages.append({
        "role": "user",
        "content": _format_user_message(question, cypher),
    })

    return messages


def _format_user_message(question: str, cypher: str) -> str:
    return f"Question: {question}\n\nReference Cypher: {cypher}"


def _format_assistant_message(reasoning: str, cypher: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"
