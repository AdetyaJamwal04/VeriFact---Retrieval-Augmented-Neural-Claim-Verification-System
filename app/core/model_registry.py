"""
Model Registry — Local NLI + Local SBERT + Optional Groq LLM

V2 Architecture:
  - NLI: cross-encoder/nli-deberta-v3-small (local, ~180MB, 88% MNLI accuracy)
  - SBERT: sentence-transformers/all-MiniLM-L6-v2 (local, ~90MB, 384-dim)
  - spaCy: en_core_web_sm (local, ~12MB, tokenization + NER)
  - LLM: Groq (optional, API-based, for claim decomposition + tiebreaker)

All heavy ML runs locally — no HF Inference API dependency.
Groq is optional and the system works without it.
"""

import os
import logging
from threading import Lock

logger = logging.getLogger(__name__)

# Thread-safe locks for singleton initialization
_locks = {
    "spacy": Lock(),
    "nli": Lock(),
    "sbert": Lock(),
    "groq": Lock(),
}

# Singleton instances
_instances = {
    "spacy": None,
    "nli_model": None,
    "nli_tokenizer": None,
    "sbert": None,
    "groq": None,
}

# Model identifiers
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

# NLI label mapping for cross-encoder models
# DeBERTa-v3 NLI outputs: [contradiction, entailment, neutral]
NLI_LABELS = ["contradiction", "entailment", "neutral"]


def get_spacy_nlp():
    """Lazy-load spaCy model (singleton)."""
    if _instances["spacy"] is None:
        with _locks["spacy"]:
            if _instances["spacy"] is None:
                import spacy
                logger.info("Loading spaCy model (en_core_web_sm)...")
                _instances["spacy"] = spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model loaded")
    return _instances["spacy"]


def get_nli_model():
    """
    Lazy-load local NLI cross-encoder model (singleton).
    
    Uses cross-encoder/nli-deberta-v3-small for direct NLI inference.
    Returns (model, tokenizer) tuple.
    """
    if _instances["nli_model"] is None:
        with _locks["nli"]:
            if _instances["nli_model"] is None:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                logger.info(f"Loading NLI model ({NLI_MODEL})...")
                tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
                model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
                model.eval()
                
                # Disable gradient computation for inference
                torch.set_grad_enabled(False)
                
                _instances["nli_model"] = model
                _instances["nli_tokenizer"] = tokenizer
                logger.info("✓ NLI model loaded (local)")
    
    return _instances["nli_model"], _instances["nli_tokenizer"]


def get_sbert_model():
    """
    Lazy-load local Sentence-BERT model (singleton).
    
    Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings.
    """
    if _instances["sbert"] is None:
        with _locks["sbert"]:
            if _instances["sbert"] is None:
                from sentence_transformers import SentenceTransformer
                
                logger.info(f"Loading SBERT model ({SBERT_MODEL})...")
                _instances["sbert"] = SentenceTransformer(SBERT_MODEL)
                logger.info("✓ SBERT model loaded (local)")
    
    return _instances["sbert"]


def nli_predict(premise: str, hypothesis: str) -> dict:
    """
    Run NLI inference locally: (premise, hypothesis) → scores.
    
    Returns dict with 'entailment', 'contradiction', 'neutral' scores (sum to 1.0).
    ~50ms per call on CPU.
    """
    import torch
    
    model, tokenizer = get_nli_model()
    
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)[0].tolist()
    
    return {
        "contradiction": scores[0],
        "entailment": scores[1],
        "neutral": scores[2],
    }


def sbert_encode(texts: list) -> "numpy.ndarray":
    """
    Encode texts into embeddings using local SBERT model.
    
    Returns numpy array of shape (len(texts), 384).
    ~10ms per sentence on CPU.
    """
    import numpy as np
    
    if not texts:
        return np.array([])
    
    model = get_sbert_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def get_groq_client():
    """
    Lazy-load Groq client (singleton). Optional — returns None if no API key.
    """
    if _instances["groq"] is None:
        with _locks["groq"]:
            if _instances["groq"] is None:
                api_key = os.getenv("GROQ_API_KEY", "")
                if not api_key:
                    logger.info("GROQ_API_KEY not set — LLM features disabled (NLI-only mode)")
                    _instances["groq"] = False  # Sentinel: tried but unavailable
                    return None
                
                try:
                    from groq import Groq
                    _instances["groq"] = Groq(api_key=api_key)
                    logger.info("✓ Groq client initialized")
                except Exception as e:
                    logger.warning(f"Groq init failed: {e}")
                    _instances["groq"] = False
                    return None
    
    if _instances["groq"] is False:
        return None
    return _instances["groq"]


def are_models_loaded() -> bool:
    """Check if core models are initialized (for health check)."""
    return all([
        _instances.get("spacy") is not None,
        _instances.get("nli_model") is not None,
        _instances.get("sbert") is not None,
    ])


def warmup_all_models():
    """Initialize all models for warmup endpoint."""
    get_spacy_nlp()
    
    # Test NLI
    nli_ok = False
    try:
        result = nli_predict("The sky is blue.", "The sky has a color.")
        nli_ok = result["entailment"] > 0.5
        logger.info(f"✓ NLI model verified (entailment: {result['entailment']:.3f})")
    except Exception as e:
        logger.warning(f"NLI warmup failed: {e}")
    
    # Test SBERT
    sbert_ok = False
    try:
        emb = sbert_encode(["test"])
        sbert_ok = emb.shape == (1, 384)
        logger.info(f"✓ SBERT model verified (dim: {emb.shape[1]})")
    except Exception as e:
        logger.warning(f"SBERT warmup failed: {e}")
    
    # Test Groq (optional)
    groq_ok = get_groq_client() is not None
    
    return {
        "spacy": _instances["spacy"] is not None,
        "nli_model": nli_ok,
        "sbert_model": sbert_ok,
        "groq_available": groq_ok,
        "nli_model_name": NLI_MODEL,
        "sbert_model_name": SBERT_MODEL,
    }
