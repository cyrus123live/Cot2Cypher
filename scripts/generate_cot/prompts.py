"""System message and prompt assembly for CoT generation."""

from generate_cot.exemplars import EXEMPLARS

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for Neo4j graph databases. \
You will be given a graph schema, a natural language question, and the correct reference Cypher query. \
Your task is to generate a step-by-step reasoning trace that explains how to arrive at the given Cypher query from the question and schema.

Follow the QDECOMP+InterCOL reasoning structure:
1. **Decompose the question** into sub-questions aligned with Cypher clause order (MATCH → WHERE → RETURN/aggregation).
2. **Link to schema elements** (InterCOL): For each sub-question, name the relevant node labels, relationship types, and properties from the schema.
3. **Identify the graph pattern**: single hop, multi-hop traversal, variable-length path, aggregation, filtering, etc.
4. **Construct the Cypher step by step**: Build the MATCH clause from identified patterns, add WHERE conditions, construct RETURN with any aggregations, ORDER BY, or LIMIT.

Output your answer in this exact format:
<reasoning>
Your step-by-step reasoning here.
</reasoning>
<cypher>
The final Cypher query here.
</cypher>

Important:
- The reasoning must reference specific node labels, relationship types, and properties FROM THE SCHEMA.
- The final Cypher in <cypher> tags must match the reference query (you may fix minor formatting but not change semantics).
- Keep reasoning concise but complete — typically 3-8 lines.
- Do not include any text outside the XML tags."""


def build_messages(
    schema: str, question: str, cypher: str
) -> list[dict[str, str]]:
    """Build the full message list: system + few-shot exemplars + target."""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    # Few-shot exemplars as alternating user/assistant pairs
    for ex in EXEMPLARS:
        messages.append({
            "role": "user",
            "content": _format_user_message(ex["schema"], ex["question"], ex["cypher"]),
        })
        messages.append({
            "role": "assistant",
            "content": _format_assistant_message(ex["reasoning"], ex["cypher"]),
        })

    # Target example
    messages.append({
        "role": "user",
        "content": _format_user_message(schema, question, cypher),
    })

    return messages


def _format_user_message(schema: str, question: str, cypher: str) -> str:
    return (
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        f"Reference Cypher: {cypher}"
    )


def _format_assistant_message(reasoning: str, cypher: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"
