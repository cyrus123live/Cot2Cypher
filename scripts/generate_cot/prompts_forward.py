"""FORWARD generation prompts — STaR-style.

Unlike prompts.py (post-hoc rationalization, where the teacher is GIVEN the
reference Cypher and asked to explain it), here the teacher sees ONLY the
schema and question and must derive the Cypher itself. The generated Cypher
is then execution-filtered: we keep a trace only if its query executes to the
same result as the reference. This produces verified-correct reasoning traces,
the ingredient the successful Text-to-SQL CoT papers (STaR-SQL) actually used.

Few-shot exemplars are reused from exemplars.py but presented in forward form:
the user turn shows schema + question only; the assistant turn shows the full
reasoning + Cypher (so the model learns the output format without ever seeing
a reference answer for the target example).
"""

from generate_cot.exemplars import EXEMPLARS

SYSTEM_MESSAGE = """\
You are an expert at translating natural language questions into Cypher queries for Neo4j graph databases. \
You will be given a graph schema and a natural language question. \
Your task is to reason step by step and then produce the correct Cypher query.

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
- Use ONLY relationship types and properties that appear in the provided schema.
- Keep reasoning concise but complete — typically 3-8 lines.
- Do not include any text outside the XML tags."""


def build_messages_forward(schema: str, question: str) -> list[dict[str, str]]:
    """Build the message list for FORWARD generation: system + few-shot + target.

    The target user turn contains schema + question ONLY (no reference Cypher).
    """
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    for ex in EXEMPLARS:
        messages.append({
            "role": "user",
            "content": _format_user_message(ex["schema"], ex["question"]),
        })
        messages.append({
            "role": "assistant",
            "content": _format_assistant_message(ex["reasoning"], ex["cypher"]),
        })

    messages.append({
        "role": "user",
        "content": _format_user_message(schema, question),
    })
    return messages


def _format_user_message(schema: str, question: str) -> str:
    return f"Schema:\n{schema}\n\nQuestion: {question}"


def _format_assistant_message(reasoning: str, cypher: str) -> str:
    return f"<reasoning>\n{reasoning}\n</reasoning>\n<cypher>\n{cypher}\n</cypher>"
