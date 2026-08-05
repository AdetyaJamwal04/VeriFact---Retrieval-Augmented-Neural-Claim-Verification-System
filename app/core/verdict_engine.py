"""
Verdict Engine Module with Accuracy Improvements

Computes the final verdict from collected evidence using weighted scoring.
Aggregates support/refute evidence and outputs confidence scores.
Includes explanation generation for transparency.

Accuracy improvements:
- Tuned decision thresholds (+0.35/-0.35 instead of ±0.4)
- Enhanced weighting formula with similarity boost
"""

import math
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Tuned thresholds for better precision/recall balance
THRESHOLD_TRUE = 0.35   # Was 0.4 - lowered for better recall
THRESHOLD_FALSE = -0.35  # Was -0.4 - raised for better recall


def sigmoid(x: float) -> float:
    """
    Sigmoid function to normalize scores to 0-1 range.
    
    Args:
        x: Input value
        
    Returns:
        Value between 0 and 1
    """
    return 1 / (1 + math.exp(-x))


def compute_weighted_score(evidence: Dict) -> Tuple[float, str]:
    """
    Calculate weighted score for a single piece of evidence.
    
    Enhanced formula: similarity × stance_score × stance_weight × source_weight × similarity_boost
    
    Args:
        evidence: Dict with 'similarity', 'stance', 'stance_score', 'source_weight'
        
    Returns:
        Tuple of (weighted_score, stance_direction)
    """
    similarity = evidence["similarity"]
    stance_score = evidence["stance_score"]
    stance = evidence["stance"]

    # Stance weight: +1 for support, -1 for refute, 0 for neutral
    if stance == "supports":
        stance_w = +1
    elif stance == "refutes":
        stance_w = -1
    else:
        stance_w = 0  # discusses / neutral
    
    source_weight = evidence.get("source_weight", 1.0)
    
    # Accuracy improvement: Boost high-similarity evidence more
    similarity_boost = 1.0 + (similarity - 0.5) * 0.5  # 0.75x to 1.25x based on similarity
    
    score = similarity * stance_score * stance_w * source_weight * similarity_boost

    return score, stance


def build_explanation(evidences: List[Dict], scores: List[Tuple[float, str]], 
                      net_score: float, verdict: str) -> Dict:
    """
    Build a structured explanation of the verdict reasoning.
    
    Args:
        evidences: List of evidence dictionaries
        scores: List of (score, stance) tuples
        net_score: Final aggregated score
        verdict: The verdict string
        
    Returns:
        Dict with 'steps', 'breakdown', and 'decision_reason'
    """
    # Count stances
    support_count = sum(1 for _, s in scores if s == "supports")
    refute_count = sum(1 for _, s in scores if s == "refutes")
    neutral_count = sum(1 for _, s in scores if s not in ("supports", "refutes"))
    
    # Calculate weights by stance
    support_weight = round(sum(sc for sc, st in scores if st == "supports"), 2)
    refute_weight = round(sum(sc for sc, st in scores if st == "refutes"), 2)
    
    # Count trusted sources
    trusted_count = sum(1 for e in evidences if e.get("source_weight", 1.0) > 1.0)
    
    # Count multi-sentence evidence
    multi_sent_count = sum(1 for e in evidences if e.get("supporting_sentences"))
    
    # Build reasoning steps
    steps = [
        {
            "step": 1,
            "title": "Evidence Collection",
            "detail": f"Found {len(evidences)} relevant source{'s' if len(evidences) != 1 else ''}",
            "icon": "🔍"
        },
        {
            "step": 2,
            "title": "Stance Analysis",
            "detail": f"{support_count} support, {refute_count} refute, {neutral_count} neutral",
            "icon": "⚖️"
        },
        {
            "step": 3,
            "title": "Credibility Weighting",
            "detail": f"{trusted_count} trusted source{'s' if trusted_count != 1 else ''} (Reuters, BBC, etc.)",
            "icon": "🏆"
        },
        {
            "step": 4,
            "title": "Score Calculation",
            "detail": f"Net score: {net_score:+.2f} (support: {support_weight:+.2f}, refute: {refute_weight:+.2f})",
            "icon": "📊"
        }
    ]
    
    # Decision reason based on verdict
    if verdict == "LIKELY TRUE":
        decision_reason = f"Score ({net_score:+.2f}) exceeds +{THRESHOLD_TRUE} threshold. The majority of credible evidence supports this claim."
    elif verdict == "LIKELY FALSE":
        decision_reason = f"Score ({net_score:+.2f}) is below {THRESHOLD_FALSE} threshold. The majority of credible evidence contradicts this claim."
    elif verdict == "UNVERIFIED":
        decision_reason = "No relevant evidence was found to verify or refute this claim."
    else:  # MIXED / MISLEADING
        decision_reason = f"Score ({net_score:+.2f}) is between {THRESHOLD_FALSE} and +{THRESHOLD_TRUE}. Evidence is conflicting or inconclusive."
    
    # Add final verdict step
    steps.append({
        "step": 5,
        "title": "Verdict",
        "detail": decision_reason,
        "icon": "✅" if verdict == "LIKELY TRUE" else "❌" if verdict == "LIKELY FALSE" else "⚠️"
    })
    
    return {
        "steps": steps,
        "breakdown": {
            "support_count": support_count,
            "refute_count": refute_count,
            "neutral_count": neutral_count,
            "support_weight": support_weight,
            "refute_weight": refute_weight,
            "trusted_sources": trusted_count,
            "total_sources": len(evidences),
            "multi_sentence_evidence": multi_sent_count
        },
        "decision_reason": decision_reason,
        "threshold_info": f"Thresholds: TRUE > +{THRESHOLD_TRUE}, FALSE < {THRESHOLD_FALSE}, MIXED in between"
    }


def compute_final_verdict(evidences: List[Dict], claim: str = "", include_explanation: bool = True) -> Dict:
    """
    Compute the final verdict based on weighted aggregation of all evidence.
    
    V2: Includes optional LLM tiebreaker for MIXED verdicts.
    
    Decision Thresholds (tuned for accuracy):
        - net_score > 0.35  → LIKELY TRUE
        - net_score < -0.35 → LIKELY FALSE
        - otherwise         → MIXED / MISLEADING (may use LLM tiebreaker)
        - no evidence       → UNVERIFIED
    
    Args:
        evidences: List of evidence dictionaries
        claim: The original claim text (needed for LLM tiebreaker)
        include_explanation: Whether to include detailed explanation
        
    Returns:
        Dict with 'verdict', 'confidence', 'net_score', and optionally 'explanation'
    """
    if not evidences:
        logger.info("No evidence found - returning UNVERIFIED")
        result = {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0,
            "summary": "No relevant evidence was found to verify or refute this claim.",
        }
        if include_explanation:
            result["explanation"] = build_explanation([], [], 0, "UNVERIFIED")
        return result

    # Compute scores with stance info
    score_data = [compute_weighted_score(e) for e in evidences]
    scores = [s[0] for s in score_data]
    net_score = sum(scores)

    confidence = sigmoid(abs(net_score))

    # Decision thresholds (tuned for better accuracy)
    if net_score > THRESHOLD_TRUE:
        verdict = "LIKELY TRUE"
    elif net_score < THRESHOLD_FALSE:
        verdict = "LIKELY FALSE"
    else:
        verdict = "MIXED / MISLEADING"
    
    # V2: LLM tiebreaker for MIXED verdicts
    # Only activates when NLI is inconclusive and Groq is available
    llm_used = False
    if verdict == "MIXED / MISLEADING" and claim:
        try:
            from app.core.llm_helper import llm_tiebreaker
            llm_result = llm_tiebreaker(claim, evidences)
            if llm_result and "verdict" in llm_result:
                llm_verdict = llm_result["verdict"]
                llm_confidence = float(llm_result.get("confidence", 0.5))
                llm_reasoning = llm_result.get("reasoning", "")
                
                # Only override MIXED if LLM is reasonably confident
                if llm_confidence >= 0.6 and llm_verdict in ("LIKELY TRUE", "LIKELY FALSE"):
                    verdict = llm_verdict
                    confidence = llm_confidence
                    llm_used = True
                    logger.info(f"LLM tiebreaker override: {verdict} ({confidence:.2f}) - {llm_reasoning}")
                elif llm_verdict == "UNVERIFIABLE":
                    verdict = "UNVERIFIED"
                    llm_used = True
                    logger.info(f"LLM tiebreaker: UNVERIFIABLE - {llm_reasoning}")
        except Exception as e:
            logger.debug(f"LLM tiebreaker skipped: {e}")

    logger.info(f"Verdict: {verdict} (score: {net_score:.3f}, confidence: {confidence:.3f}{', LLM-assisted' if llm_used else ''})")

    result = {
        "verdict": verdict,
        "confidence": round(confidence, 3),
        "net_score": round(net_score, 3),
        "llm_assisted": llm_used,
    }
    
    if include_explanation:
        result["explanation"] = build_explanation(evidences, score_data, net_score, verdict)
    
    # Generate plain-language LLM summary
    try:
        from app.core.llm_helper import generate_verdict_summary
        result["summary"] = generate_verdict_summary(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            evidence_list=evidences,
            explanation=result.get("explanation"),
        )
    except Exception as e:
        logger.debug(f"LLM summary generation skipped: {e}")
        # Fallback to rule-based decision reason
        if result.get("explanation") and "decision_reason" in result["explanation"]:
            result["summary"] = result["explanation"]["decision_reason"]
        else:
            result["summary"] = f"The claim was assessed as {verdict}."
    
    return result
