"""Parse and validate LLM responses with XML tags."""

import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    reasoning: str
    cypher: str
    success: bool
    error: str = ""


def parse_response(text: str) -> ParseResult:
    """Extract <reasoning> and <cypher> from LLM response.

    Tries XML tags first, then falls back to heuristic parsing.
    """
    reasoning = _extract_tag(text, "reasoning")
    cypher = _extract_tag(text, "cypher")

    if reasoning and cypher:
        return ParseResult(
            reasoning=reasoning.strip(),
            cypher=_clean_cypher(cypher.strip()),
            success=True,
        )

    # Fallback: try to find reasoning and cypher without proper tags
    if not reasoning and not cypher:
        return _fallback_parse(text)

    # Partial: one tag found but not the other
    if reasoning and not cypher:
        # Try to find cypher after the reasoning section
        after_reasoning = text.split("</reasoning>")[-1] if "</reasoning>" in text else ""
        cypher = after_reasoning.strip()
        if cypher:
            return ParseResult(
                reasoning=reasoning.strip(),
                cypher=_clean_cypher(cypher),
                success=True,
            )

    if cypher and not reasoning:
        # Try to find reasoning before the cypher section
        before_cypher = text.split("<cypher>")[0] if "<cypher>" in text else ""
        reasoning = before_cypher.strip()
        if reasoning:
            return ParseResult(
                reasoning=reasoning,
                cypher=_clean_cypher(cypher.strip()),
                success=True,
            )

    return ParseResult(
        reasoning=reasoning or "",
        cypher=cypher or "",
        success=False,
        error="Could not extract both reasoning and cypher",
    )


def _extract_tag(text: str, tag: str) -> str | None:
    """Extract content between XML tags."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def _clean_cypher(cypher: str) -> str:
    """Remove markdown code fences and extra whitespace from cypher."""
    cypher = cypher.strip()
    # Remove ```cypher ... ``` wrapping
    cypher = re.sub(r"^```(?:cypher)?\s*\n?", "", cypher)
    cypher = re.sub(r"\n?```\s*$", "", cypher)
    cypher = cypher.strip()
    return cypher


def _fallback_parse(text: str) -> ParseResult:
    """Attempt to parse response without proper XML tags."""
    # Look for "Reasoning:" / "Cypher:" markers
    reasoning_match = re.search(
        r"(?:reasoning|thinking|explanation)\s*:\s*(.*?)(?=(?:cypher|query)\s*:|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    cypher_match = re.search(
        r"(?:cypher|query)\s*(?:output)?\s*:\s*(.*?)$",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    if reasoning_match and cypher_match:
        return ParseResult(
            reasoning=reasoning_match.group(1).strip(),
            cypher=_clean_cypher(cypher_match.group(1).strip()),
            success=True,
        )

    return ParseResult(
        reasoning="",
        cypher="",
        success=False,
        error="No XML tags or recognizable markers found",
    )


def validate_result(result: ParseResult) -> ParseResult:
    """Validate parsed result has non-empty, reasonable content."""
    if not result.success:
        return result

    if len(result.reasoning) < 20:
        return ParseResult(
            reasoning=result.reasoning,
            cypher=result.cypher,
            success=False,
            error=f"Reasoning too short ({len(result.reasoning)} chars)",
        )

    if len(result.cypher) < 5:
        return ParseResult(
            reasoning=result.reasoning,
            cypher=result.cypher,
            success=False,
            error=f"Cypher too short ({len(result.cypher)} chars)",
        )

    # Basic Cypher sanity: should contain MATCH or RETURN or CALL
    cypher_upper = result.cypher.upper()
    if not any(kw in cypher_upper for kw in ("MATCH", "RETURN", "CALL", "CREATE", "MERGE")):
        return ParseResult(
            reasoning=result.reasoning,
            cypher=result.cypher,
            success=False,
            error="Cypher missing expected keywords (MATCH/RETURN/CALL/CREATE/MERGE)",
        )

    return result
