# VeriFact - Retrieval Augmented Neural Claim Verification System
# Technical Implementation Report

## 1. Introduction

### 1.1 Problem Statement
The rapid dissemination of information across digital platforms has exacerbated the spread of misinformation. Manual verification of claims is labor-intensive and cannot scale to meet the volume of modern content creation. There is a critical need for automated systems capable of verifying textual claims against credible external evidence in near real-time.

### 1.2 Motivation
This project, **VeriFact**, aims to bridge the gap between information retrieval and semantic reasoning. Instead of relying on static databases of previously debunked claims, the system implements a dynamic verification pipeline that retrieves live evidence from the web and uses Natural Language Inference (NLI) to determine veracity.

### 1.3 Scope and Boundaries
The system is designed as an **inference-only** pipeline. It does not train new foundation models but rather orchestrates pre-trained Transformer models to perform specific reasoning tasks. Its scope is limited to:
- Validating textual claims against English-language web sources.
- Proving provenance and semantic alignment of evidence.
- Deterministically aggregating logic-based signals into a verdict.
It does not verify image/video content (deepfakes) or generate new knowledge beyond what is retrieved.

---

## 2. System Overview

### 2.1 High-Level Description
VeriFact is a modular, retrieval-augmented claim verification API. It accepts natural language claims, retrieves relevant documents from the open web, filters content using semantic similarity, and analyzes the stance of the retrieved content relative to the claim. The final output is a structured verdict ("Likely True", "Likely False", etc.) supported by a confidence score, a transparent audit trail of evidence, and an optional AI-generated plain-language summary.

### 2.2 Design Philosophy
The system prioritizes **provenance** and **explainability** over generative capabilities. Unlike Large Language Models (LLMs) which may hallucinate facts, VeriFact functions as a neural search-and-verify engine. Every component of the verdict is traceable to a specific source URL and a specific sentence within that source. The LLM (Groq/Llama-3.3-70b) is used only as an optional enhancement — for query decomposition, tiebreaking, and summary generation — never as the primary decision-maker.

### 2.3 Data Flow
1.  **Input Ingestion:** Raw text/URL is received via API.
2.  **Claim Extraction:** Core assertion is isolated from input noise using sentence importance scoring.
3.  **Query Generation:** Search queries are generated via LLM decomposition (primary) or NER-based expansion (fallback).
4.  **Information Retrieval:** Queries are executed against a 3-tier search fallback chain (Tavily → Brave → DuckDuckGo).
5.  **Evidence Processing:** Articles are scraped, segmented, and filtered by semantic similarity using SBERT embeddings.
6.  **Stance Detection:** Surviving content is analyzed via local cross-encoder NLI (DeBERTa-v3-small), with pre-filters for outcome mismatches and relationship claims.
7.  **Aggregation:** Signals are weighted by similarity, confidence, stance direction, source credibility, and similarity boost, then fused into a final score.
8.  **Verdict & Summary:** A verdict is computed with optional LLM tiebreaker for MIXED cases, and an AI-generated plain-language summary is produced.

---

## 3. Architecture and System Design

### 3.1 Architectural Components
The system follows a layered architecture:
-   **Interface Layer:** Flask REST API with Pydantic input validation, rate limiting (5/min per IP, 100/hour), CORS support, and WhiteNoise for production static file serving. Served via Gunicorn with threaded workers.
-   **Controller Layer:** The `check_claim()` function in `app_flask.py` orchestrates the end-to-end pipeline. Synchronous at the request level but internally parallelizes I/O-bound work.
-   **Core Services Layer:** 11 specialized modules, each with a single responsibility:
    -   `claim_extractor` — Input normalization and sentence importance scoring
    -   `query_generator` — Dual-strategy query expansion (LLM + NER fallback)
    -   `web_search` — Multi-API retrieval with 3-tier fallback chain
    -   `scraper` — `trafilatura`-based content extraction with connection pooling
    -   `embedder` — SBERT-based semantic similarity with LRU caching
    -   `stance_detector` — Direct NLI via local cross-encoder, with outcome mismatch detection and relationship claim verification
    -   `evidence_aggregator` — Parallel scraping, embedding, and stance aggregation via ThreadPoolExecutor
    -   `verdict_engine` — Weighted scoring, verdict computation, and LLM tiebreaker
    -   `source_scorer` — Three-tier domain credibility weighting
    -   `llm_helper` — Optional Groq-powered claim decomposition, tiebreaking, and AI summary generation
    -   `model_registry` — Thread-safe singleton management for all local ML models
-   **External Interfaces:** Wrappers for Tavily, Brave, DuckDuckGo search APIs, Groq LLM API, and HuggingFace model hub (for model downloads only).

### 3.2 Interaction and Control Flow
Execution is synchronous but parallelized at the I/O layer. The main thread accepts a request and invokes the pipeline. The **Evidence Aggregation** phase utilizes a `ThreadPoolExecutor` (3 workers) to perform concurrent web scraping and processing of multiple search results, significantly reducing total latency.

### 3.3 Failure Handling
-   **Search Redundancy:** The search module implements a 3-tier fallback chain (Tavily → Brave → DuckDuckGo) to ensure continuity if primary APIs fail or are rate-limited. DuckDuckGo further falls back to news search if web search returns empty.
-   **Graceful Degradation:** If specific URLs fail to scrape (403, Cloudflare, paywall), they are logged and skipped without halting the pipeline. If Groq LLM is unavailable, the system falls back to NER-based queries, rule-based explanations, and skips the tiebreaker.
-   **Safety Defaults:** If no evidence is found, the system defaults to an "UNVERIFIED" state rather than guessing.

---

## 4. Implementation Details

### 4.1 Repository Structure
The codebase is organized as follows:
-   `app_flask.py`: Application entry point and API definition.
-   `app/core/`: Functional logic (11 modules).
    -   `claim_extractor.py`: Input normalization using spaCy sentence scoring + local SBERT embedding-based keyword extraction.
    -   `query_generator.py`: Dual-strategy query expansion (LLM decomposition + NER fallback).
    -   `web_search.py`: 3-tier search API integration (Tavily → Brave → DuckDuckGo).
    -   `scraper.py`: `trafilatura`-based HTML text extraction with connection pooling and retry logic.
    -   `embedder.py`: Local SBERT embedding logic with LRU caching.
    -   `evidence_aggregator.py`: Parallel evidence processing with ThreadPoolExecutor.
    -   `stance_detector.py`: Direct NLI cross-encoder pipeline with outcome mismatch detection (v2.2) and relationship claim verification (v2.3).
    -   `verdict_engine.py`: Weighted scoring algorithms with LLM tiebreaker and AI summary.
    -   `source_scorer.py`: Three-tier domain credibility weighting.
    -   `llm_helper.py`: Groq LLM integration for query decomposition, tiebreaking, and summaries.
    -   `model_registry.py`: Thread-safe singleton model management with lazy loading and warmup.
-   `Dockerfile`: Container definition.
-   `docker-compose.yml`: Docker Compose configuration.
-   `.github/workflows/deploy.yml`: CI/CD configuration.

### 4.2 Entry Points
The application runs via a **Gunicorn** WSGI server in production. The `app_flask.py` module initializes the Flask app, configures rate limiting and CORS, and exposes four endpoints: `/` (Web UI), `/api/health` (health check), `/api/check` (claim verification), and `/api/warmup` (model pre-loading). Heavy ML models are lazy-loaded on first request via the `model_registry` singleton pattern.

---

## 5. NLP and Machine Learning Pipeline

### 5.1 Claim Processing
Raw inputs are cleaned to normalize whitespace. If a URL is provided, `trafilatura` extracts the main text body. The system uses a **sentence importance scoring** heuristic to identify the primary claim:
-   **Named entity density:** +2 points per unique entity type (PERSON, ORG, GPE, etc.)
-   **Statistical content:** +1 if the sentence contains numeric tokens
-   **Length constraints:** +1 for sentences between 30–200 characters (optimal for verification)
-   **Quotation marks:** +1.5 (direct quotes are strong claim indicators)
-   **Position bonus:** Decaying bonus for earlier sentences (first +1.5, second +1.0, third +0.5)

Additionally, the system runs **embedding-based keyword extraction** using the local SBERT model (`all-MiniLM-L6-v2`). It generates n-gram candidates from noun chunks and named entities, embeds them alongside the full text, ranks by cosine similarity, and applies **Maximal Marginal Relevance (MMR)** with λ=0.7 to balance relevance and diversity. These keywords feed downstream query generation.

### 5.2 Query Generation
The system uses a **dual-strategy approach** to generate search queries:

**Strategy A — LLM Decomposition (Primary):** When Groq is available, the claim is sent to Llama-3.3-70b with a structured prompt requesting 4–5 precise, verifiable search queries. The LLM is instructed to include queries that could *disprove* the claim — critical for avoiding confirmation bias.

**Strategy B — NER-based Expansion (Fallback):** Using spaCy's `en_core_web_sm`, named entities are extracted. The system generates:
-   The exact claim + verification suffixes ("fact check", "true or false", "hoax")
-   Entity-centric queries ("*entity* controversy", "*entity* news verification")
-   Keyword-based queries from the MMR-selected keywords

Queries are capped at 10 and deduplicated. The LLM path preserves order via `dict.fromkeys()`; the NER fallback deduplicates via `set()`.

### 5.3 Evidence Retrieval and Cleaning
URLs are retrieved via a **3-tier search fallback chain**:
1.  **Tavily** (primary) — `advanced` search depth, best quality, fastest. Requires API key.
2.  **Brave Search** (secondary) — 2,000 free requests/month, reliable. 0.3-second delay between queries.
3.  **DuckDuckGo** (tertiary) — Free, no API key. Uses `lite` backend to avoid rate limiting, with 3 retries and exponential backoff. Falls back to DDG news search if web search returns empty.

API clients are **cached as singletons** to avoid TCP connection overhead. Results are URL-deduplicated across all tiers.

Retrieved URLs are scraped using `trafilatura` with an 8-second timeout and connection pooling (10 connections). The `requests.Session` implements automatic retries (2 retries with 0.5s backoff for 429/5xx errors) and a custom User-Agent. If a URL returns 403 (Cloudflare, paywall), it is gracefully skipped.

Extracted text (capped at 10,000 characters) is segmented into sentences using spaCy's sentence boundary detection, filtered to >20 characters, and capped at 50 sentences per source.

### 5.4 Embedding Generation
The system uses **Sentence-BERT (SBERT)** with `all-MiniLM-L6-v2` running locally to map the claim ($C$) and every candidate sentence ($S_i$) into a **384-dimensional vector space**. Cosine similarity is computed:
$$Similarity(C, S_i) = \frac{C \cdot S_i}{\|C\| \|S_i\|}$$

The claim embedding is **LRU-cached** (32 entries), so if multiple sources are compared against the same claim, the claim is embedded only once.

For each source, the sentence with the highest similarity score is retained for stance analysis, acting as a relevance filter. Sentences below a **similarity threshold of 0.25** are early-exited — they skip stance detection entirely and are marked as "discusses" with 0 confidence.

### 5.5 Stance Detection Logic
The system uses a **cross-encoder NLI model: `nli-deberta-v3-small`** running locally. The cross-encoder takes the evidence sentence as the **premise** and the claim as the **hypothesis**, and outputs three logits: *entailment*, *contradiction*, *neutral*. These are softmaxed into probabilities.

**Why a Cross-Encoder and Not a Bi-Encoder?**
A bi-encoder (like SBERT) encodes premise and hypothesis separately — fast but loses cross-attention between the two texts. A cross-encoder processes the pair jointly through all Transformer layers, enabling it to capture fine-grained logical relationships. For NLI, cross-encoders consistently outperform bi-encoders by 5–8% on benchmarks.

**Why DeBERTa-v3-small (Direct NLI) and Not BART-MNLI (Zero-Shot)?**
The original system used `facebook/bart-large-mnli` via HuggingFace's Inference API as a zero-shot classifier. Migration to DeBERTa-v3 as a direct cross-encoder NLI model was driven by three factors:
1.  **Architectural correctness:** Zero-shot classification repurposes NLI by constructing synthetic hypotheses. Direct NLI takes the actual premise-hypothesis pair — this is fundamentally what stance detection requires.
2.  **Reliability:** External API calls introduced rate limits, 503 errors, and unpredictable latency. Local inference eliminates all of that.
3.  **Latency control:** Local inference at ~50ms/call on CPU is deterministic, versus 200ms–2s from an API.

**Confidence Calibration:** Raw softmax outputs from neural networks are overconfident. Temperature scaling with T=1.3 is applied:
$$calibrated = raw^{1/T}$$

**Minimum Confidence Thresholds:** Entailment ("supports") requires ≥0.50 confidence; contradiction ("refutes") requires ≥0.40. The asymmetry is intentional — false support (topic overlap) is more common than false refutation.

**High-Stakes Claim Handling:** Claims containing keywords like "killed," "assassinated," "nuclear," "war" activate a stricter threshold: support requires ≥0.70 confidence.

### 5.6 Outcome Mismatch Detection (v2.2)
A pre-filter module runs **before** NLI inference to catch cases where SBERT similarity is misleadingly high but the semantic meaning is opposite.

**Detection pipeline:**
-   **Negating modifiers:** "attempted," "survived," "failed," "thwarted," "prevented" in evidence but not in claim → override to "refutes" (0.75 confidence)
-   **Action changes:** "repositioned," "planned," "threatened" → override to "discusses" (0.60 confidence)
-   **Explicit negation patterns:** Regex patterns for "not dead," "is alive," "hoax," "debunked," "misinformation," "conspiracy theory"

**Example:** Claim "Trump was assassinated" + Evidence "survived assassination attempt" → mismatch detected → "refutes" (0.75).

### 5.7 Relationship Claim Verification (v2.3)
Claims asserting familial relationships require special handling because NLI models are poor at entity-level factual verification.

**Detection Pipeline:**
1.  `is_relationship_claim()` — Keyword scan for "son of," "daughter of," "married to," etc.
2.  `extract_relationship_entities()` — Regex-based extraction of (subject, relationship, object) tuples.
3.  `verify_relationship_claim()` — If evidence mentions the same subject with the same relationship type but a *different* object → "refutes" with 0.85 confidence.

**Example:** Claim "Rahul Gandhi is Modi's son" + Evidence "Rahul Gandhi is the son of Rajiv Gandhi" → contradiction detected → "refutes" (0.85).

For relationship claims, the minimum confidence threshold for "supports" is raised to 0.80 to prevent false positives from topical overlap.

---

## 6. Model Selection and Usage

### 6.1 Models Used
1.  **`cross-encoder/nli-deberta-v3-small`:** Used for direct NLI stance detection. A cross-encoder model trained on the Multi-Genre NLI (MNLI) corpus. Achieves 88% MNLI accuracy while being lightweight (~180MB) and efficient on CPU (~50ms/pair).
2.  **`sentence-transformers/all-MiniLM-L6-v2`:** Used for semantic similarity and keyword extraction. Produces 384-dimensional embeddings. Excellent performance on STS benchmarks (~78% Spearman) while being compact (~90MB) and fast on CPU (~10ms/sentence).
3.  **`en_core_web_sm`:** spaCy's lightweight CPU-optimized model (~12MB) for tokenization, sentence segmentation, and NER.
4.  **`llama-3.3-70b-versatile`:** API-based LLM hosted on Groq. Used optionally for query decomposition, verdict tiebreaking, and AI summary generation (~1–2s per call).

### 6.2 Inference Methodology
The system uses **Direct NLI (Cross-Encoder) inference** rather than zero-shot classification. This is a crucial design choice: the cross-encoder takes the evidence-claim pair and directly classifies the logical relationship (entailment/contradiction/neutral). Unlike zero-shot classifiers that construct synthetic hypotheses, direct NLI preserves the original semantic relationship.

The NLI model was trained on MNLI, which teaches generalized *logical reasoning* — it understands entailment and contradiction at a linguistic level, not at a topic level. This makes the system **domain-agnostic** and future-proof without any custom training.

### 6.3 Thread-Safe Model Management
All models are managed through a centralized `model_registry` with:
-   **Double-checked locking:** Thread-safe singleton initialization using `threading.Lock` for each model.
-   **Lazy loading:** Models are only loaded when first needed, saving ~3 minutes of startup time.
-   **Warmup endpoint:** `/api/warmup` POST endpoint triggers pre-loading of all models with verification tests.
-   **Gradient disabled:** `torch.set_grad_enabled(False)` globally for inference, reducing memory footprint.
-   **Groq sentinel pattern:** Groq client uses a `False` sentinel to distinguish "not yet tried" from "tried but unavailable."

### 6.4 Inference Trade-offs
-   **Latency:** Transformer inference on CPU is computationally expensive. The system manages this by limiting input length, utilizing threading for web I/O, and caching claim embeddings.
-   **Throughput:** Single-instance throughput is limited. Production scaling would require GPU acceleration or horizontal scaling.
-   **Memory:** Running DeBERTa + MiniLM + spaCy simultaneously approaches the 4 GB limit. Workers are limited to 3, sentences capped at 50 per source, and spaCy processing capped at 10K characters.

---

## 7. Evidence Aggregation and Verdict Logic

### 7.1 Evidence Scoring
Each piece of evidence $E$ is assigned a scalar score based on five factors:
1.  **Similarity ($Sim$):** How relevant the text is (0.0 – 1.0).
2.  **Stance Confidence ($Conf$):** How sure the model is of the relationship (0.0 – 1.0), after temperature calibration.
3.  **Direction ($Dir$):** +1 (Support), –1 (Refute), 0 (Neutral/Discusses).
4.  **Source Credibility ($W_{src}$):** Domain-based weight (0.3 – 1.5).
5.  **Similarity Boost ($Boost$):** $1.0 + (Sim - 0.5) \times 0.5$ — amplifies high-similarity evidence (up to 1.25×), dampens low-similarity evidence (down to 0.75×).

$$Score_i = Sim_i \times Conf_i \times Dir_i \times W_{src,i} \times Boost_i$$

### 7.2 Credibility Weighting
The `source_scorer` module applies a domain-based weight ($W_{src}$) to each evidence source:
-   **Tier 1 (Wire services, fact-checkers):** Reuters, AP, Snopes, PolitiFact ($W = 1.5$).
-   **Tier 1.5 (Major newspapers, gov/edu):** BBC, NYT, Guardian, `.gov`, `.edu` ($W = 1.3 – 1.4$).
-   **Tier 2 (Standard/Unknown):** General news, unlisted domains ($W = 1.0$).
-   **Tier 3 (Social media):** Twitter, Facebook, Reddit, TikTok ($W = 0.3 – 0.6$).

This is a static lookup table — a known limitation. New or unlisted high-quality sources receive the default weight, while unlisted low-quality sources are not penalized.

### 7.3 Verdict Computation
The final **Net Score** is the sum of weighted evidence signals:
$$NetScore = \sum_{i=1}^{N} Score_i$$

### 7.4 Verdict Thresholds
The Net Score is mapped to a discrete verdict (thresholds tuned from ±0.4 to ±0.35 for better recall):
-   **Likely True:** Score $> 0.35$
-   **Likely False:** Score $< -0.35$
-   **Mixed / Misleading:** $-0.35 \le Score \le 0.35$
-   **Unverified:** No evidence found.

Confidence is derived via sigmoid: $confidence = \sigma(|NetScore|) = \frac{1}{1 + e^{-|NetScore|}}$

### 7.5 LLM Tiebreaker (v2.0)
When the NLI pipeline produces a MIXED verdict, the system optionally invokes Groq's Llama-3.3-70b as a tiebreaker. The LLM receives the claim and up to 8 pieces of evidence with their stances, and synthesizes a verdict with reasoning.

The LLM's verdict is only accepted if:
-   Confidence ≥ 0.60
-   Verdict is LIKELY TRUE, LIKELY FALSE, or UNVERIFIABLE

The NLI verdict always takes priority when decisive. The LLM is a fallback, not the primary decision-maker.

### 7.6 AI-Generated Summary
The system generates a 2–3 sentence plain-language summary explaining *why* the verdict was reached, using Groq with a prompt that forbids technical jargon, scoring references, or system-centric language. If Groq is unavailable, the system falls back to a rule-based decision reason from the explanation builder.

---

## 8. API Design

### 8.1 Endpoint Definitions
-   `GET /`: Serves the web UI.
-   `GET /api/health`: System status, uptime, model load state, and basic metrics.
-   `POST /api/check`: The primary claim verification endpoint.
-   `POST /api/warmup`: Pre-loads all ML models to avoid cold-start latency.

### 8.2 Request/Response Contract
**Request:**
```json
{
  "claim": "string",
  "text": "optional string",
  "url": "optional string",
  "max_results": "int (1-10, default 3)"
}
```

**Response:**
```json
{
  "claim": "string",
  "verdict": "LIKELY TRUE | LIKELY FALSE | MIXED / MISLEADING | UNVERIFIED",
  "confidence": 0.85,
  "net_score": 1.25,
  "summary": "Plain-language explanation of the verdict",
  "explanation": { "steps": [...], "breakdown": {...}, "decision_reason": "..." },
  "evidences": [ ... ],
  "sources_analyzed": 3,
  "processing_time": 12.5,
  "status": "success"
}
```

### 8.3 Error Handling
All exceptions are caught at the controller level. Input validation uses Pydantic with field validators for `max_results` (1–10) and `url` (must start with http/https). Rate limiting (5/min per IP, 100/hour) uses in-memory storage. Global error handlers standardize responses into JSON format (400/429/500), ensuring the API never exposes raw stack traces.

---

## 9. Deployment and Infrastructure

### 9.1 Containerization
The application is packaged in a **Docker** container based on `python:3.10-slim`. The base image includes only essential system dependencies (`gcc`, `g++`, `curl`). A non-root user (`appuser`) is created for security. ML models are lazy-loaded at runtime (downloaded on first request or via `/api/warmup`), keeping the base image lightweight.

### 9.2 Runtime Environment
-   **Server:** AWS EC2 (Ubuntu).
-   **Process Manager:** Gunicorn with `gthread` workers — 2 workers × 4 threads per worker — to handle concurrent I/O requests alongside blocking CPU inference.
-   **Timeout:** Configured to 120 seconds to accommodate worst-case CPU inference.
-   **Worker recycling:** `--max-requests 500` with jitter of 50 for periodic garbage collection.
-   **Static files:** WhiteNoise middleware serves static assets in production.
-   **Health check:** Docker HEALTHCHECK at 30-second intervals with 30-second start period.

### 9.3 CI/CD Workflow
GitHub Actions orchestrates the deployment:
1.  **Test:** Runs `pytest` suite.
2.  **Build:** Builds Docker image and pushes to DockerHub.
3.  **Deploy:** SSHs into EC2 instance, performs disk cleanup (pruning images/containers), pulls the new image, and restarts the container.

---

## 10. Limitations

### 10.1 Technical Limitations
-   **CPU Inference Latency:** Running heavy Transformers (DeBERTa/SBERT) on standard CPUs results in high latency (8–18s per request end-to-end).
-   **Context Window:** The system extracts the single "best" sentence per source. Complex claims requiring multi-sentence reasoning or cross-document context may be misclassified.
-   **Memory Constraints:** Running three Transformer models on a 4 GB instance requires careful memory management — reduced thread pools, capped evidence, and disabled gradients.

### 10.2 Model Limitations
-   **NLI Fallibility:** While robust, the cross-encoder NLI model can misinterpret sarcasm, subtle nuances, or claims that are technically correct but misleading.
-   **Entity-Level Verification:** NLI models understand logical relationships but not factual correctness of specific entity attributes. The relationship claim module (v2.3) partially addresses this with regex-based entity extraction.
-   **Language Support:** The current pipeline is optimized for English; performance on other languages is undefined.

### 10.3 Data Limitations
-   **Source Reputation:** Domain credibility is based on a static lookup table. New or unlisted high-quality sources receive a default weight, while unlisted low-quality sources are not penalized.
-   **Search Coverage:** Claims about events not yet indexed by search engines will return UNVERIFIED.
-   **Scraping Limitations:** Paywalled or Cloudflare-protected sites reduce the available evidence pool.

---

## 11. Conclusion

VeriFact implements a robust, transparent, and extendable architecture for automated fact-checking. By combining modern Neural Information Retrieval (SBERT) with Natural Language Inference (DeBERTa-v3 cross-encoder), enhanced by pre-filters for outcome mismatches and relationship claims, it moves beyond simple keyword matching to perform true semantic verification. The inference-only design ensures strict adherence to retrieved evidence, minimizing hallucination risks and providing users with a verifiable audit trail for every verdict. The optional LLM integration (Groq/Llama-3.3-70b) enhances query quality and provides plain-language summaries without compromising the system's core deterministic reasoning.
