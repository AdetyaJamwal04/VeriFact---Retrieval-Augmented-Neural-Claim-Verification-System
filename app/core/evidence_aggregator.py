"""
Evidence Aggregator Module with Accuracy Improvements

Aggregates evidence from search results by scraping, embedding, and stance detection.

Accuracy improvements applied:
- Multi-sentence evidence (top-3 sentences per source)
- Reduced spaCy processing limit 50k→10k
- Sentence limit for SBERT encoding
"""

from app.core.scraper import scrape_article
from app.core.embedder import get_best_matching_sentences
from app.core.stance_detector import detect_stance
from app.core.source_scorer import get_source_weight, is_social_media
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

# Similarity threshold for early exit (skip stance detection for very low similarity)
SIMILARITY_THRESHOLD = 0.25  # Lowered slightly for more coverage

# Number of top sentences to analyze per source (accuracy improvement)
TOP_SENTENCES_PER_SOURCE = 1  # V2: reduced from 3 — best match is sufficient with local NLI


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences using spacy or fallback to simple splitting.
    
    Args:
        text: Article text to split
        
    Returns:
        List of sentences with length > 20 chars
    """
    from app.core.model_registry import get_spacy_nlp
    
    try:
        nlp = get_spacy_nlp()
        # Reduced from 50000 to 10000 chars
        doc = nlp(text[:10000])
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        # Limit to first 50 sentences
        return sentences[:50]
    except Exception:
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
        return sentences[:50]


def process_single_result(item: dict, claim: str) -> dict:
    """
    Process a single search result to extract evidence.
    Uses multi-sentence evidence for higher accuracy.
    
    Args:
        item: Search result dict with 'href', 'title', 'body'
        claim: The claim being fact-checked
        
    Returns:
        Evidence dict or None if processing fails
    """
    try:
        url = item.get("href")
        if not url:
            return None
            
        article_text = scrape_article(url)
        if not article_text:
            return None
            
        sentences = split_into_sentences(article_text)
        if not sentences:
            return None
        
        # Get top N matching sentences (accuracy improvement)
        top_sentences = get_best_matching_sentences(claim, sentences, top_n=TOP_SENTENCES_PER_SOURCE)
        
        if not top_sentences:
            return None
        
        # Use the best sentence for primary evidence
        best_sentence, best_sim = top_sentences[0]
        
        # Early exit for very low similarity
        if best_sim < SIMILARITY_THRESHOLD:
            logger.debug(f"Skipping stance detection for low-sim ({best_sim:.2f}): {url}")
            return {
                "url": url,
                "best_sentence": best_sentence,
                "similarity": best_sim,
                "stance": "discusses",
                "stance_score": 0.0,
                "source_weight": get_source_weight(url),
                "is_social_media": is_social_media(url),
                "supporting_sentences": [],
            }
        
        # Perform stance detection on all top sentences for accuracy
        stance_results = []
        for sent, sim in top_sentences:
            if sim >= SIMILARITY_THRESHOLD:
                stance = detect_stance(sent, claim)
                stance_results.append({
                    "sentence": sent,
                    "similarity": sim,
                    "stance": stance["label"],
                    "confidence": stance["confidence"]
                })
        
        # Aggregate stance from multiple sentences
        aggregated_stance = aggregate_sentence_stances(stance_results)
        
        source_weight = get_source_weight(url)
        
        # Debug: log per-source stance results
        stance_summary = ", ".join([f"{r['stance']}({r['confidence']:.2f})" for r in stance_results])
        logger.info(f"Source {url[:50]}... sim={best_sim:.3f} stances=[{stance_summary}] → {aggregated_stance['label']}({aggregated_stance['confidence']:.2f}) weight={source_weight:.1f}")
        
        return {
            "url": url,
            "best_sentence": best_sentence,
            "similarity": best_sim,
            "stance": aggregated_stance["label"],
            "stance_score": aggregated_stance["confidence"],
            "source_weight": source_weight,
            "is_social_media": is_social_media(url),
            "supporting_sentences": stance_results[1:] if len(stance_results) > 1 else [],
        }
    except Exception as e:
        logger.warning(f"Error processing {item.get('href', 'unknown')}: {e}")
        return None


def aggregate_sentence_stances(stance_results: list) -> dict:
    """
    Aggregate stance results from multiple sentences into a single stance.
    Uses weighted voting based on similarity and confidence.
    
    Args:
        stance_results: List of dicts with 'sentence', 'similarity', 'stance', 'confidence'
        
    Returns:
        dict with aggregated 'label' and 'confidence'
    """
    if not stance_results:
        return {"label": "discusses", "confidence": 0.0}
    
    if len(stance_results) == 1:
        return {"label": stance_results[0]["stance"], "confidence": stance_results[0]["confidence"]}
    
    # Weight by similarity × confidence
    support_score = 0.0
    refute_score = 0.0
    neutral_score = 0.0
    
    for result in stance_results:
        weight = result["similarity"] * result["confidence"]
        if result["stance"] == "supports":
            support_score += weight
        elif result["stance"] == "refutes":
            refute_score += weight
        else:
            neutral_score += weight
    
    total = support_score + refute_score + neutral_score
    if total == 0:
        return {"label": "discusses", "confidence": 0.0}
    
    # Determine winning stance
    if support_score >= refute_score and support_score >= neutral_score:
        label = "supports"
        confidence = support_score / total
    elif refute_score >= support_score and refute_score >= neutral_score:
        label = "refutes"
        confidence = refute_score / total
    else:
        label = "discusses"
        confidence = neutral_score / total
    
    return {"label": label, "confidence": confidence}


def build_evidence(claim: str, search_results: list, max_workers: int = 3):
    """
    Build evidence from search results with parallel processing.
    Reduced max_workers from 5 to 3 to limit thread memory overhead.
    """
    evidences = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_result, item, claim): item 
            for item in search_results
        }
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                evidences.append(result)
    
    return evidences
