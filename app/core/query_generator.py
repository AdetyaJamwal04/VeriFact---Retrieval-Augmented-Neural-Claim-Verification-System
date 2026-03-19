"""
Query Generator Module — V2 with LLM Enhancement

Generates search queries from a claim using:
1. LLM decomposition (Groq, if available) — precise, diverse queries
2. NER-based fallback — keyword-based queries using spaCy entities

Falls back gracefully when LLM is unavailable.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def generate_queries(claim: str, keywords: list[str] | None = None) -> List[str]:
    """
    Generate multiple search queries from a claim.
    
    V2: Tries LLM decomposition first for more precise, diverse queries.
    Falls back to NER-based queries if LLM is unavailable.
    
    Args:
        claim: The claim to generate queries for
        keywords: Optional list of keywords extracted by KeyBERT
        
    Returns:
        List of unique search query strings
    """
    # Try LLM decomposition first (much better queries)
    try:
        from app.core.llm_helper import decompose_claim
        llm_queries = decompose_claim(claim)
        if llm_queries and len(llm_queries) >= 3:
            # Always include the raw claim + fact check as baseline
            all_queries = [claim.lower(), f"{claim.lower()} fact check"] + llm_queries
            unique = list(dict.fromkeys(all_queries))  # Dedup preserving order
            logger.info(f"Generated {len(unique)} queries (LLM + baseline)")
            return unique[:10]
    except Exception as e:
        logger.debug(f"LLM query generation failed: {e}")
    
    # Fallback: NER-based queries
    return _ner_based_queries(claim, keywords)


def _ner_based_queries(claim: str, keywords: list[str] | None = None) -> List[str]:
    """
    Generate queries using Named Entity Recognition (original method).
    """
    from app.core.model_registry import get_spacy_nlp
    
    nlp = get_spacy_nlp()
    doc = nlp(claim)
    entities = [ent.text for ent in doc.ents]
    
    logger.debug(f"Extracted entities: {entities}")
    if keywords:
        logger.debug(f"Using keywords: {keywords}")

    base = claim.lower()

    queries = [
        base,
        base + " fact check",
        base + " true or false",
        base + " hoax",
        base + " authenticity check",
    ]

    # Entity-based queries for better coverage
    for e in entities:
        queries.append(f"{e} {base}")
        queries.append(f"{base} {e} false")
        queries.append(f"{e} controversy")
        queries.append(f"{e} news verification")

    # Keyword-based queries (from KeyBERT)
    if keywords:
        for kw in keywords[:3]:
            queries.append(f"{kw} fact check")
            queries.append(f"{kw} {base}")
            queries.append(f"{kw} news")

    unique_queries = list(set(queries))
    unique_queries = unique_queries[:10]
    logger.info(f"Generated {len(unique_queries)} queries from claim (NER fallback)")
    
    return unique_queries
