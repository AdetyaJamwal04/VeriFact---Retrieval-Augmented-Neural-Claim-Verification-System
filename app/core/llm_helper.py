"""
LLM Helper — Groq Integration (Optional Enhancement)

Provides two LLM-powered features that enhance accuracy without replacing the NLI pipeline:
1. Claim decomposition → better search queries
2. Verdict tiebreaker → resolves MIXED verdicts

Falls back gracefully when Groq API key is not set — system works without it.
"""

import json
import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


def _call_groq(prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> Optional[str]:
    """
    Call Groq API with a prompt. Returns response text or None on failure.
    """
    from app.core.model_registry import get_groq_client, GROQ_MODEL
    
    client = get_groq_client()
    if client is None:
        return None
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq API error: {e}")
        return None


def decompose_claim(claim: str) -> Optional[List[str]]:
    """
    Use LLM to decompose a claim into precise, verifiable search queries.
    
    Example:
        "Elon Musk lives on Mars" → [
            "does Elon Musk live on Mars",
            "where does Elon Musk currently reside",
            "has any human traveled to Mars",
            "Elon Musk Mars residence fact check"
        ]
    
    Returns None if Groq is unavailable (caller falls back to NER-based queries).
    """
    prompt = f"""You are a fact-checking assistant. Given a claim, generate 4-5 precise search queries 
that would help verify whether the claim is true or false.

IMPORTANT RULES:
- Focus on finding EVIDENCE, not just the claim text
- Include queries that could DISPROVE the claim
- Include a fact-check specific query
- Be specific and search-engine friendly
- Return ONLY a JSON array of strings, nothing else

Claim: "{claim}"

JSON array of search queries:"""
    
    response = _call_groq(prompt, max_tokens=300)
    if response is None:
        return None
    
    try:
        # Parse JSON array from response
        # Handle cases where LLM wraps in markdown code block
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # Remove first line
            text = text.rsplit("```", 1)[0]  # Remove last ```
            text = text.strip()
        
        queries = json.loads(text)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info(f"LLM decomposed claim into {len(queries)} queries")
            return queries[:6]  # Cap at 6
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM queries: {e}")
    
    return None


def llm_tiebreaker(claim: str, evidence_list: List[Dict]) -> Optional[Dict]:
    """
    Use LLM to break a tie when NLI gives a MIXED verdict.
    
    Takes the claim and all collected evidence, asks the LLM to synthesize
    a final verdict with reasoning.
    
    Args:
        claim: The original claim
        evidence_list: List of dicts with 'url', 'best_sentence', 'stance', etc.
    
    Returns:
        dict with 'verdict', 'confidence', 'reasoning' or None if unavailable
    """
    if not evidence_list:
        return None
    
    # Build evidence summary for the prompt
    evidence_text = ""
    for i, ev in enumerate(evidence_list[:8], 1):  # Max 8 sources
        sentence = ev.get("best_sentence", "")
        stance = ev.get("stance", "unknown")
        url = ev.get("url", "unknown")
        if sentence:
            evidence_text += f"\n{i}. [{stance}] \"{sentence}\" (Source: {url})\n"
    
    if not evidence_text.strip():
        return None
    
    prompt = f"""You are a fact-checking assistant. Based on the evidence below, determine if the claim is TRUE, FALSE, or UNVERIFIABLE.

CLAIM: "{claim}"

EVIDENCE FROM MULTIPLE SOURCES:
{evidence_text}

Analyze the evidence carefully. Consider:
1. Do any sources DIRECTLY confirm or deny the claim?
2. Is the evidence about a RELATED topic but doesn't actually confirm the claim?
3. What is the overall weight of evidence?

Respond in this EXACT JSON format:
{{"verdict": "LIKELY TRUE" or "LIKELY FALSE" or "UNVERIFIABLE", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}}

JSON response:"""
    
    response = _call_groq(prompt, max_tokens=300, temperature=0.1)
    if response is None:
        return None
    
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()
        
        result = json.loads(text)
        if "verdict" in result and "confidence" in result:
            logger.info(f"LLM tiebreaker: {result['verdict']} ({result['confidence']:.2f})")
            return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse LLM tiebreaker: {e}")
    
    return None


def generate_verdict_summary(
    claim: str,
    verdict: str,
    confidence: float,
    evidence_list: List[Dict],
    explanation: Optional[Dict] = None,
) -> str:
    """
    Generate a plain-language summary explaining why the verdict was reached.

    Uses the LLM to produce 2-3 human-friendly sentences that explain the
    verdict in terms a non-technical user can understand — focusing on the
    evidence and sources, NOT the scoring math.

    Falls back to the rule-based decision_reason when Groq is unavailable.

    Args:
        claim: The original claim text
        verdict: The computed verdict string (e.g. "LIKELY TRUE")
        confidence: Confidence score 0-1
        evidence_list: List of evidence dicts with 'url', 'best_sentence', 'stance', etc.
        explanation: Optional structured explanation dict from build_explanation()

    Returns:
        str: Plain-language summary (2-3 sentences)
    """
    # Build fallback from existing rule-based explanation
    fallback = ""
    if explanation and "decision_reason" in explanation:
        fallback = explanation["decision_reason"]
    elif not evidence_list:
        fallback = "No relevant evidence was found to verify or refute this claim."
    else:
        fallback = f"The claim was assessed as {verdict} based on {len(evidence_list)} source(s)."

    # Try LLM-powered summary
    if not evidence_list:
        return fallback

    # Build a concise evidence summary for the prompt
    evidence_summary = ""
    for i, ev in enumerate(evidence_list[:6], 1):
        sentence = ev.get("best_sentence", "")
        stance = ev.get("stance", "unknown")
        url = ev.get("url", "")
        weight = ev.get("source_weight", 1.0)

        # Extract hostname for readability
        source_name = url
        try:
            from urllib.parse import urlparse
            source_name = urlparse(url).hostname or url
            if source_name.startswith("www."):
                source_name = source_name[4:]
        except Exception:
            pass

        trust_label = "trusted" if weight > 1.0 else "social media" if weight < 1.0 else "standard"
        evidence_summary += f"{i}. [{stance}] \"{sentence}\" — {source_name} ({trust_label})\n"

    # Breakdown stats
    breakdown_info = ""
    if explanation and "breakdown" in explanation:
        bd = explanation["breakdown"]
        breakdown_info = (
            f"Evidence breakdown: {bd.get('support_count', 0)} supporting, "
            f"{bd.get('refute_count', 0)} refuting, {bd.get('neutral_count', 0)} neutral. "
            f"{bd.get('trusted_sources', 0)} from trusted sources."
        )

    prompt = f"""You are a fact-checking assistant. Write a clear, plain-language summary (2-3 sentences) explaining WHY the following claim received its verdict.

CLAIM: "{claim}"
VERDICT: {verdict} (confidence: {confidence:.0%})

EVIDENCE ANALYZED:
{evidence_summary}
{breakdown_info}

RULES:
- Explain in simple terms a non-technical person can understand
- Reference specific sources or evidence that drove the verdict
- Do NOT mention scores, thresholds, algorithms, or technical details
- Do NOT use phrases like "based on our analysis" or "the system determined"
- Write as if you are a journalist explaining the finding
- Keep it to 2-3 sentences maximum
- Be direct and factual

Plain-language summary:"""

    response = _call_groq(prompt, max_tokens=200, temperature=0.3)
    if response:
        summary = response.strip()
        # Remove quotes if the LLM wrapped the response
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        logger.info(f"LLM verdict summary generated ({len(summary)} chars)")
        return summary

    logger.debug("LLM summary unavailable, using rule-based fallback")
    return fallback


def is_available() -> bool:
    """Check if Groq LLM is available."""
    from app.core.model_registry import get_groq_client
    return get_groq_client() is not None
