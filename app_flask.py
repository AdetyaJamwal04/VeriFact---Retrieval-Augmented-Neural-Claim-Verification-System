import os
os.environ["USE_TF"] = "NO"       # Tell transformers to skip TensorFlow entirely (we use PyTorch only)
os.environ["USE_TORCH"] = "YES"   # Explicitly prefer PyTorch
"""
Flask REST API for Claim Verification with Optimizations

Features:
- Rate limiting (5 requests/minute per IP)
- Input validation with Pydantic
- Health check with metrics
- Comprehensive error handling

Optimizations applied:
- Warmup endpoint (Rank 11)
- Fixed health check model status (Rank 12)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, field_validator, ValidationError
from whitenoise import WhiteNoise
from typing import Optional
import sys
import time
import logging
import threading
from datetime import datetime

# Load .env file at startup (must be before other app imports)
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.claim_extractor import extract_text_from_url, extract_claim_from_text
from app.core.query_generator import generate_queries
from app.core.web_search import web_search
from app.core.evidence_aggregator import build_evidence
from app.core.verdict_engine import compute_final_verdict

# Initialize Flask
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')
CORS(app)

# Wrap with WhiteNoise for production static file serving (handles Content-Length properly)
app.wsgi_app = WhiteNoise(app.wsgi_app, root=STATIC_DIR, prefix='static/')

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe Metrics
START_TIME = datetime.now()
_metrics_lock = threading.Lock()
REQUEST_COUNT = 0
ERROR_COUNT = 0


def increment_request_count():
    """Thread-safe increment of request counter."""
    global REQUEST_COUNT
    with _metrics_lock:
        REQUEST_COUNT += 1


def increment_error_count():
    """Thread-safe increment of error counter."""
    global ERROR_COUNT
    with _metrics_lock:
        ERROR_COUNT += 1


# Input validation model
class CheckRequest(BaseModel):
    """Pydantic model for input validation."""
    text: Optional[str] = None
    url: Optional[str] = None
    claim: Optional[str] = None
    max_results: int = 3
    
    @field_validator('max_results')
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError('max_results must be between 1 and 10')
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory('static', 'index.html')


@app.route('/api')
def api_info():
    """API information endpoint."""
    return jsonify({
        'name': 'VeriFact - Retrieval Augmented Neural Claim Verification System',
        'version': '2.1.0',
        'status': 'running',
        'endpoints': {
            '/api/health': 'Health check with metrics',
            '/api/check': 'Fact-check a claim (POST)',
            '/api/warmup': 'Preload models (POST)'
        }
    })


@app.route('/api/health')
def health_check():
    """Health check with metrics and actual model status."""
    from app.core.model_registry import are_models_loaded
    
    uptime = (datetime.now() - START_TIME).total_seconds()
    
    # Optimization #12: Reflect actual model load state
    models_loaded = are_models_loaded()
    
    return jsonify({
        'status': 'healthy',
        'version': '2.1.0',
        'uptime_seconds': round(uptime, 2),
        'requests_processed': REQUEST_COUNT,
        'errors': ERROR_COUNT,
        'models_loaded': models_loaded
    })


# Optimization #11: Warmup endpoint
@app.route('/api/warmup', methods=['POST'])
def warmup():
    """
    Preload all ML models to avoid cold-start latency on first request.
    Call this after deployment to warm up the container.
    """
    from app.core.model_registry import warmup_all_models
    
    start_time = time.time()
    logger.info("Starting model warmup...")
    
    try:
        status = warmup_all_models()
        warmup_time = round(time.time() - start_time, 2)
        logger.info(f"Model warmup completed in {warmup_time}s")
        
        return jsonify({
            'status': 'warm',
            'models': status,
            'warmup_time_seconds': warmup_time
        })
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/check', methods=['POST'])
@limiter.limit("5 per minute")
def check_claim():
    """
    Fact-check a claim.
    
    Request body (JSON):
        - claim: Direct claim text
        - text: Article text to extract claim from
        - url: URL to extract claim from
        - max_results: Number of evidence sources (1-10)
    
    Returns:
        - verdict: LIKELY TRUE | LIKELY FALSE | MIXED | UNVERIFIED
        - confidence: 0-1 confidence score
        - evidences: List of evidence sources
    """
    increment_request_count()
    start_time = time.time()
    
    try:
        # Parse and validate input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request body must be JSON'}), 400
        
        try:
            validated = CheckRequest(**data)
        except ValidationError as e:
            return jsonify({
                'error': 'Validation error',
                'details': e.errors(include_url=False, include_context=False)
            }), 400
        
        # Check that at least one input is provided
        if not validated.text and not validated.url and not validated.claim:
            return jsonify({
                'error': 'Must provide at least one of: text, url, or claim'
            }), 400
        
        # Extract claim
        keywords = []
        if validated.claim:
            claim = validated.claim.strip()
        elif validated.url:
            logger.info(f"Extracting from URL: {validated.url}")
            full_text = extract_text_from_url(validated.url)
            if not full_text:
                return jsonify({'error': 'Could not extract text from URL'}), 400
            claim, keywords = extract_claim_from_text(full_text)
        else:
            claim, keywords = extract_claim_from_text(validated.text)
        
        if not claim or len(claim) < 10:
            return jsonify({'error': 'Could not extract a valid claim'}), 400
        
        logger.info(f"Processing claim: {claim[:100]}...")
        if keywords:
            logger.info(f"Extracted keywords: {keywords}")
        
        # Process pipeline - include keywords for enhanced query generation
        queries = generate_queries(claim, keywords=keywords)
        search_results = web_search(queries, max_results=validated.max_results)
        evidences = build_evidence(claim, search_results)
        verdict_result = compute_final_verdict(evidences, claim=claim)
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Completed in {processing_time}s - Verdict: {verdict_result['verdict']}")
        
        # Response
        return jsonify({
            'claim': claim,
            'verdict': verdict_result['verdict'],
            'confidence': verdict_result['confidence'],
            'net_score': verdict_result['net_score'],
            'summary': verdict_result.get('summary', ''),
            'explanation': verdict_result.get('explanation'),
            'evidences': evidences,
            'sources_analyzed': len(evidences),
            'processing_time': processing_time,
            'status': 'success'
        })
        
    except Exception as e:
        increment_error_count()
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e) if app.debug else 'An error occurred',
            'status': 'error'
        }), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded errors."""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.',
        'retry_after': e.description
    }), 429


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, threaded=True, debug=debug)
