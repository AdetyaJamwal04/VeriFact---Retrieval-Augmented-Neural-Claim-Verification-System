"""
Claim Extractor Module

Extracts claims from URLs or text using NLP techniques.
Uses spaCy for sentence segmentation and HF Inference API
for keyword extraction (replaces KeyBERT with same algorithm).

Models are lazy-loaded via model_registry.
"""

from trafilatura import fetch_url, extract
import nltk
import logging
import numpy as np
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# Download required NLTK data (lightweight, ~5MB)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


def extract_text_from_url(url: str) -> str:
    """
    Fetch and clean article content from a URL.
    
    Args:
        url: The URL of the article to extract text from
        
    Returns:
        Cleaned article text, or empty string if extraction fails
    """
    try:
        html = fetch_url(url)
        if not html:
            logger.warning(f"Failed to fetch HTML from {url}")
            return ""
        text = extract(html)
        return text or ""
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return ""


def clean_text(text: str) -> str:
    """Normalize whitespace and formatting."""
    if not text:
        return ""
    return " ".join(text.split())


def score_sentence_importance(sentence: str, doc) -> float:
    """
    Score sentence importance based on multiple factors.
    
    Args:
        sentence: The sentence to score
        doc: The spacy doc containing entities
        
    Returns:
        Importance score (higher = more important)
    """
    from app.core.model_registry import get_spacy_nlp
    nlp = get_spacy_nlp()
    
    score = 0.0
    
    # Contains named entities (+2 per entity type)
    sentence_ents = [ent for ent in doc.ents if ent.text in sentence]
    score += len(set(ent.label_ for ent in sentence_ents)) * 2.0
    
    # Contains numbers/statistics (+1)
    sentence_doc = nlp(sentence)
    if any(tok.like_num for tok in sentence_doc):
        score += 1.0
    
    # Good length: not too short, not too long (+1)
    if 30 < len(sentence) < 200:
        score += 1.0
    
    # Contains quotation marks (likely a claim) (+1.5)
    if '"' in sentence or "'" in sentence:
        score += 1.5
    
    return score


def extract_keywords_hf(text: str, top_n: int = 5) -> List[str]:
    """
    Extract keywords using HF API embeddings — same algorithm as KeyBERT.
    
    Algorithm:
    1. Generate n-gram candidates from the text
    2. Embed the full text and each candidate via SBERT (HF API)
    3. Rank candidates by cosine similarity to the text
    4. Apply MMR (Maximal Marginal Relevance) for diversity
    
    Args:
        text: The text to extract keywords from
        top_n: Number of keywords to return
        
    Returns:
        List of keyword strings
    """
    from app.core.model_registry import get_spacy_nlp
    from app.core.embedder import _embed_texts, _cosine_similarity
    
    if not text or len(text.strip()) < 10:
        return []
    
    nlp = get_spacy_nlp()
    doc = nlp(text[:2000])  # Limit to avoid excessive processing
    
    # Generate candidates: noun chunks + named entities + significant tokens
    candidates = set()
    
    # Noun chunks (1-2 word phrases)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip().lower()
        if len(chunk_text) > 2 and not chunk.root.is_stop:
            candidates.add(chunk_text)
    
    # Named entities
    for ent in doc.ents:
        ent_text = ent.text.strip().lower()
        if len(ent_text) > 2:
            candidates.add(ent_text)
    
    # Important single tokens (nouns, proper nouns, adjectives)
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "ADJ") and not token.is_stop and len(token.text) > 2:
            candidates.add(token.text.lower())
    
    if not candidates:
        return []
    
    candidate_list = list(candidates)[:30]  # Cap candidates
    
    # Embed text and candidates via HF API
    try:
        all_texts = [text] + candidate_list
        embeddings = _embed_texts(all_texts)
        
        if embeddings.size == 0 or len(embeddings) < 2:
            return []
        
        text_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Score candidates by similarity to full text
        scores = []
        for idx, candidate in enumerate(candidate_list):
            if idx < len(candidate_embeddings):
                sim = _cosine_similarity(text_embedding, candidate_embeddings[idx])
                scores.append((candidate, sim, idx))
        
        # MMR: balance relevance with diversity
        selected = []
        remaining = list(scores)
        
        for _ in range(min(top_n, len(remaining))):
            if not remaining:
                break
            
            if not selected:
                # First keyword: highest similarity to text
                best = max(remaining, key=lambda x: x[1])
            else:
                # Subsequent: balance relevance and diversity
                lambda_param = 0.7  # 0.7 relevance, 0.3 diversity
                best_score = -1
                best = remaining[0]
                
                for candidate_text, sim_to_text, idx in remaining:
                    # Max similarity to already-selected keywords
                    max_sim_to_selected = max(
                        _cosine_similarity(candidate_embeddings[idx], candidate_embeddings[s_idx])
                        for _, _, s_idx in selected
                    )
                    mmr_score = lambda_param * sim_to_text - (1 - lambda_param) * max_sim_to_selected
                    
                    if mmr_score > best_score:
                        best_score = mmr_score
                        best = (candidate_text, sim_to_text, idx)
            
            selected.append(best)
            remaining.remove(best)
        
        return [kw[0] for kw in selected]
        
    except Exception as e:
        logger.warning(f"HF keyword extraction failed: {e}")
        # Fallback: use spaCy noun chunks (still good, just not embedding-based)
        fallback_keywords = []
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and len(chunk.text.strip()) > 2:
                fallback_keywords.append(chunk.text.strip().lower())
            if len(fallback_keywords) >= top_n:
                break
        return fallback_keywords


def extract_claim_from_text(text: str) -> Tuple[str, List[str]]:
    """
    Extracts the main claim by choosing the most important sentence.
    Uses sentence importance scoring instead of always taking the first sentence.
    Also extracts keywords using embedding-based extraction for enhanced query generation.
    
    Args:
        text: The article text to extract claim from
        
    Returns:
        Tuple of (claim_sentence, keywords_list)
    """
    from app.core.model_registry import get_spacy_nlp
    
    text = clean_text(text)
    if not text:
        return "", []

    nlp = get_spacy_nlp()
    doc = nlp(text[:5000])
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

    if not sentences:
        return text[:500], []

    # Score each sentence for importance
    scored_sentences = []
    for idx, sent in enumerate(sentences[:10]):
        score = score_sentence_importance(sent, doc)
        position_bonus = max(0, 3 - idx) * 0.5
        total_score = score + position_bonus
        scored_sentences.append((sent, total_score))
    
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    claim = scored_sentences[0][0]

    # Extract keywords using HF API embeddings (same accuracy as KeyBERT)
    try:
        keywords = extract_keywords_hf(claim, top_n=5)
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        keywords = []

    return claim, keywords