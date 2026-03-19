"""
Stance Detector Module with Optimizations

Uses local cross-encoder NLI (nli-deberta-v3-small) to determine if evidence
supports, refutes, or discusses a claim via direct natural language inference.

Model: cross-encoder/nli-deberta-v3-small (local, ~180MB, 88% MNLI accuracy).

Optimizations applied:
- Confidence calibration via temperature scaling (Rank 13)
- Outcome modifier detection (v2.2 - fixes false positives)
- Relationship claim verification (v2.3)
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Outcome modifiers that indicate non-completion or different action
OUTCOME_MODIFIERS = {
    "negating": [
        "attempted", "attempt", "tried", "try", "failed", "almost", "nearly",
        "survived", "survive", "escaped", "avoided", "thwarted", "foiled",
        "unsuccessful", "aborted", "prevented", "stopped", "blocked"
    ],
    "action_change": [
        "repositioned", "reposition", "moved", "deployed", "positioned",
        "threatened", "warned", "considered", "planned", "proposed"
    ]
}

# High-stakes claim keywords that require stricter checking
HIGH_STAKES_KEYWORDS = [
    "killed", "dead", "died", "assassinated", "assassination",
    "attack", "attacked", "war", "nuclear", "bomb", "explosion",
    "arrested", "convicted", "resigned", "impeached"
]

# Relationship claim patterns that require explicit entity verification
RELATIONSHIP_PATTERNS = [
    r"(\w+)\s+is\s+(\w+)'s\s+(son|daughter|father|mother|wife|husband|brother|sister|child|parent)",
    r"(\w+)\s+is\s+the\s+(son|daughter|father|mother|wife|husband|brother|sister|child|parent)\s+of\s+(\w+)",
    r"(\w+)\s+married\s+to\s+(\w+)",
    r"(\w+)\s+born\s+to\s+(\w+)",
]

# Relationship keywords for quick detection
RELATIONSHIP_KEYWORDS = [
    "son of", "daughter of", "father of", "mother of", 
    "wife of", "husband of", "brother of", "sister of",
    "child of", "parent of", "married to", 
    "'s son", "'s daughter", "'s father", "'s mother", "'s wife", "'s husband",
    "s son", "s daughter", "s father", "s mother", "s wife", "s husband",  # Without apostrophe
    "is the son", "is the daughter", "is the father", "is the mother",
    "is the wife", "is the husband", "is the child", "is the parent"
]


def get_current_model() -> str:
    """Return the currently used model name."""
    from app.core.model_registry import NLI_MODEL
    return NLI_MODEL


# Optimization #13: Confidence calibration
def calibrate_confidence(raw_score: float, temperature: float = 1.3) -> float:
    """Apply temperature scaling for better calibrated confidence scores."""
    return raw_score ** (1 / temperature)


def is_high_stakes_claim(claim: str) -> bool:
    """Check if the claim involves high-stakes assertions requiring stricter verification."""
    claim_lower = claim.lower()
    return any(keyword in claim_lower for keyword in HIGH_STAKES_KEYWORDS)


def is_relationship_claim(claim: str) -> bool:
    """Check if the claim asserts a familial/personal relationship between entities."""
    claim_lower = claim.lower()
    return any(keyword in claim_lower for keyword in RELATIONSHIP_KEYWORDS)


def extract_relationship_entities(text: str) -> Tuple[str, str, str]:
    """
    Extract relationship entities from text: (person1, relationship, person2).
    
    Examples:
    - "Rahul Gandhi is Narendra Modi's son" -> ("rahul gandhi", "son", "narendra modi")
    - "X is the daughter of Y" -> ("x", "daughter", "y")
    
    Returns:
        Tuple of (subject, relationship_type, object) or (None, None, None)
    """
    text_lower = text.lower()
    
    # Pattern 1: "X is Y's [relationship]" (with apostrophe)
    pattern1 = r"(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)'s\s+(son|daughter|father|mother|wife|husband|brother|sister|child|parent)"
    match = re.search(pattern1, text_lower)
    if match:
        return (match.group(1).strip(), match.group(3).strip(), match.group(2).strip())
    
    # Pattern 2: "X is the [relationship] of Y"
    pattern2 = r"(\w+(?:\s+\w+)?)\s+is\s+the\s+(son|daughter|father|mother|wife|husband|brother|sister|child|parent)\s+of\s+(\w+(?:\s+\w+)?)"
    match = re.search(pattern2, text_lower)
    if match:
        return (match.group(1).strip(), match.group(2).strip(), match.group(3).strip())
    
    # Pattern 3: "X is Ys [relationship]" (possessive without apostrophe, e.g., "Rahul is Modis son")
    pattern3 = r"(\w+(?:\s+\w+)?)\s+is\s+(\w+(?:\s+\w+)?)s\s+(son|daughter|father|mother|wife|husband|brother|sister|child|parent)"
    match = re.search(pattern3, text_lower)
    if match:
        return (match.group(1).strip(), match.group(3).strip(), match.group(2).strip())
    
    return (None, None, None)


def verify_relationship_claim(claim: str, evidence: str) -> Tuple[bool, str]:
    """
    Verify if evidence supports/refutes a relationship claim.
    
    For relationship claims like "X is Y's son", we need to check if:
    1. Evidence mentions a DIFFERENT relationship for the same person
    2. Evidence explicitly names a different parent/relative
    
    Returns:
        Tuple of (has_contradiction, reason)
    """
    claim_entities = extract_relationship_entities(claim)
    evidence_entities = extract_relationship_entities(evidence)
    
    if claim_entities[0] is None:
        return (False, "")
    
    claim_subject, claim_rel, claim_object = claim_entities
    
    # If evidence also mentions a relationship for the same subject
    if evidence_entities[0] is not None:
        ev_subject, ev_rel, ev_object = evidence_entities
        
        # Same subject, same relationship type, but DIFFERENT object = contradiction
        if claim_subject in ev_subject or ev_subject in claim_subject:
            if claim_rel == ev_rel and claim_object not in ev_object and ev_object not in claim_object:
                return (True, f"different_{claim_rel}:{ev_object}")
    
    # Check for explicit naming patterns that contradict the claim
    evidence_lower = evidence.lower()
    
    # If claim says "X is Y's son" but evidence says "X is the son of Z" (Z != Y)
    if claim_rel in ["son", "daughter", "child"]:
        # Look for "son/daughter of [someone]" in evidence
        parent_pattern = rf"{claim_subject}\s+is\s+the\s+{claim_rel}\s+of\s+(\w+(?:\s+\w+)?)"
        match = re.search(parent_pattern, evidence_lower)
        if match:
            actual_parent = match.group(1).strip()
            if claim_object not in actual_parent and actual_parent not in claim_object:
                return (True, f"actual_parent:{actual_parent}")
    
    return (False, "")


def detect_outcome_mismatch(claim: str, evidence: str) -> Tuple[bool, str]:
    """
    Detect if evidence describes an attempted/modified action while claim asserts completion.
    
    Examples:
    - Claim: "Trump was assassinated" + Evidence: "survived assassination attempt" → mismatch
    - Claim: "ordered attack" + Evidence: "ordered repositioning" → mismatch
    
    Returns:
        Tuple of (has_mismatch, reason)
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()
    
    # Check for negating modifiers in evidence but not in claim
    for modifier in OUTCOME_MODIFIERS["negating"]:
        if modifier in evidence_lower and modifier not in claim_lower:
            # Evidence says "attempted/survived/failed" but claim doesn't mention it
            return True, f"outcome_negated:{modifier}"
    
    # Check for action changes
    for modifier in OUTCOME_MODIFIERS["action_change"]:
        if modifier in evidence_lower:
            # Evidence describes a different action (repositioned vs attacked)
            # Only flag if claim doesn't contain this action
            if modifier not in claim_lower:
                return True, f"action_changed:{modifier}"
    
    # Check for explicit negation patterns
    negation_patterns = [
        r"not\s+dead",
        r"is\s+alive",
        r"survived",
        r"did\s+not\s+(die|attack|kill|bomb)",
        r"no\s+(attack|assassination|death)",
        r"false\s+claim",
        r"debunked",
        r"hoax",
        r"misinformation",
        r"conspiracy\s+theor"
    ]
    
    for pattern in negation_patterns:
        if re.search(pattern, evidence_lower):
            if "hoax" not in claim_lower and "false" not in claim_lower:
                return True, f"explicit_negation:{pattern}"
    
    return False, ""


def detect_stance(evidence_sentence: str, claim: str) -> Dict:
    """
    Performs direct NLI stance detection using local cross-encoder model.
    
    Uses natural language inference to determine if the evidence
    supports, refutes, or is neutral towards the claim.
    
    V2: Uses local cross-encoder (nli-deberta-v3-small) instead of HF API.
    Direct NLI is fundamentally more accurate than zero-shot classification.
    
    Args:
        evidence_sentence: The sentence from the article (premise)
        claim: The claim to fact-check (hypothesis)
    
    Returns:
        dict: {label: str, confidence: float, modifier_detected: bool}
    """
    from app.core.model_registry import nli_predict
    
    if not evidence_sentence or not claim:
        return {"label": "neutral", "confidence": 0, "modifier_detected": False}
    
    premise = evidence_sentence.strip()
    claim = claim.strip()
    
    # Check for outcome mismatch BEFORE running NLI
    has_mismatch, mismatch_reason = detect_outcome_mismatch(claim, premise)
    
    if has_mismatch:
        logger.info(f"Outcome mismatch detected: {mismatch_reason}")
        if "negated" in mismatch_reason or "explicit_negation" in mismatch_reason:
            return {
                "label": "refutes",
                "confidence": 0.75,
                "modifier_detected": True,
                "mismatch_reason": mismatch_reason
            }
        else:
            return {
                "label": "discusses",
                "confidence": 0.6,
                "modifier_detected": True,
                "mismatch_reason": mismatch_reason
            }
    
    # v2.3: Check for relationship claim verification
    if is_relationship_claim(claim):
        has_contradiction, contradiction_reason = verify_relationship_claim(claim, premise)
        if has_contradiction:
            logger.info(f"Relationship contradiction detected: {contradiction_reason}")
            return {
                "label": "refutes",
                "confidence": 0.85,
                "modifier_detected": True,
                "mismatch_reason": f"relationship_mismatch:{contradiction_reason}"
            }

    try:
        # V2: Direct NLI inference — (premise, hypothesis=claim)
        # Cross-encoder outputs: {entailment, contradiction, neutral} scores
        # This directly answers: "Does this evidence support/contradict the claim?"
        nli_scores = nli_predict(premise, claim)
        
        entailment = nli_scores["entailment"]
        contradiction = nli_scores["contradiction"]
        neutral = nli_scores["neutral"]
        
        # Map NLI labels to stance labels
        if entailment > contradiction and entailment > neutral:
            label = "supports"
            raw_confidence = entailment
        elif contradiction > entailment and contradiction > neutral:
            label = "refutes"
            raw_confidence = contradiction
        else:
            label = "discusses"
            raw_confidence = neutral
        
        # Minimum confidence thresholds
        # Entailment needs higher threshold (topic overlap causes false positives)
        # Contradiction needs lower threshold (harder to detect but still meaningful)
        MIN_SUPPORT_CONFIDENCE = 0.50
        MIN_REFUTE_CONFIDENCE = 0.40
        if label == "supports" and raw_confidence < MIN_SUPPORT_CONFIDENCE:
            logger.debug(f"Low entailment ({raw_confidence:.3f}) - downgrading to discusses")
            label = "discusses"
        elif label == "refutes" and raw_confidence < MIN_REFUTE_CONFIDENCE:
            logger.debug(f"Low contradiction ({raw_confidence:.3f}) - downgrading to discusses")
            label = "discusses"
        
        # Apply confidence calibration
        confidence = calibrate_confidence(raw_confidence)
        
        # For high-stakes claims, require higher confidence for "supports"
        if is_high_stakes_claim(claim) and label == "supports":
            if confidence < 0.7:
                label = "discusses"
                logger.info(f"High-stakes claim, low confidence ({confidence:.2f}) - downgrading to discusses")
        
        # For relationship claims, require VERY high confidence for "supports"
        if is_relationship_claim(claim) and label == "supports":
            if confidence < 0.8:
                label = "discusses"
                logger.info(f"Relationship claim, insufficient confidence ({confidence:.2f}) - downgrading to discusses")
        
        if label == "neutral":
            label = "discusses"
        
        return {
            "label": label,
            "confidence": confidence,
            "modifier_detected": False
        }
    except Exception as e:
        logger.error(f"Stance detection error: {e}")
        return {"label": "discusses", "confidence": 0, "modifier_detected": False}


def batch_detect_stance(premises: List[str], claim: str) -> List[Dict]:
    """
    Batch stance detection for multiple premises.
    
    Args:
        premises: List of evidence sentences
        claim: The claim to check against
        
    Returns:
        List of stance results
    """
    return [detect_stance(premise, claim) for premise in premises]

