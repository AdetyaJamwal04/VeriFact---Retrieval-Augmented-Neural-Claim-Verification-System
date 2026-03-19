"""
Semantic Similarity Module — Local SBERT

Uses sentence-transformers/all-MiniLM-L6-v2 running locally for
embedding generation and cosine similarity computation.

V2: Runs locally instead of HF Inference API — no timeouts, no rate limits.
"""

import logging
import numpy as np
from functools import lru_cache
from typing import List, Tuple

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed texts using local SBERT model.
    
    Args:
        texts: List of strings to embed.
        
    Returns:
        numpy array of shape (len(texts), 384).
    """
    from app.core.model_registry import sbert_encode
    
    if not texts:
        return np.array([])
    
    try:
        return sbert_encode(texts)
    except Exception as e:
        logger.error(f"SBERT embedding error: {e}")
        return np.zeros((len(texts), 384))


@lru_cache(maxsize=32)
def _embed_claim(claim: str) -> tuple:
    """Cache claim embeddings (converted to tuple for hashability)."""
    embedding = _embed_texts([claim])
    if embedding.size == 0:
        return tuple([0.0] * 384)
    return tuple(embedding[0].tolist())


def get_best_matching_sentences(
    claim: str, 
    sentences: List[str],
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """
    Find top-N sentences most semantically similar to the claim.
    
    Uses local SBERT embeddings and cosine similarity.
    
    Args:
        claim: The claim to compare against
        sentences: List of candidate sentences
        top_n: Number of top matches to return
    
    Returns:
        List of (sentence, similarity_score) tuples, sorted by score descending
    """
    if not sentences or not claim:
        return []
    
    # Filter out very short sentences
    valid_sentences = [s for s in sentences if len(s.strip()) > 20]
    if not valid_sentences:
        return []
    
    # Get claim embedding (cached)
    claim_vec = np.array(_embed_claim(claim))
    
    # Get sentence embeddings (batch — fast locally)
    sentence_vecs = _embed_texts(valid_sentences)
    
    if sentence_vecs.size == 0:
        return []
    
    # Compute cosine similarities
    scored_sentences = []
    for idx, sent in enumerate(valid_sentences):
        if idx < len(sentence_vecs):
            sim = _cosine_similarity(claim_vec, sentence_vecs[idx])
            scored_sentences.append((sent, float(sim)))
    
    # Sort by similarity (descending) and return top N
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return scored_sentences[:top_n]
