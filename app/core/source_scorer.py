"""
Source Credibility Scorer

Assigns credibility weights to sources based on domain.
Social media gets lower weight, trusted news sources get higher weight.
"""

from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

# Trusted news sources - higher credibility (1.3-1.5)
TRUSTED_SOURCES = {
    # Major wire services
    "reuters.com": 1.5,
    "apnews.com": 1.5,
    "afp.com": 1.5,
    
    # Major newspapers
    "bbc.com": 1.4,
    "bbc.co.uk": 1.4,
    "nytimes.com": 1.4,
    "washingtonpost.com": 1.4,
    "theguardian.com": 1.4,
    "economist.com": 1.4,
    
    # Fact-checking sites
    "snopes.com": 1.5,
    "factcheck.org": 1.5,
    "politifact.com": 1.5,
    "fullfact.org": 1.5,
    
    # Academic/Government
    "edu": 1.3,  # Educational domains
    "gov": 1.4,  # Government domains
    "gov.in": 1.4,
    "nic.in": 1.4,
    
    # India-specific trusted sources
    "thehindu.com": 1.3,
    "indianexpress.com": 1.3,
    "hindustantimes.com": 1.3,
    "ndtv.com": 1.2,
    "timesofindia.com": 1.2,
}

# Social media - lower credibility (0.4-0.6)
SOCIAL_MEDIA_SOURCES = {
    "twitter.com": 0.5,
    "x.com": 0.5,
    "facebook.com": 0.4,
    "instagram.com": 0.4,
    "reddit.com": 0.6,  # Reddit slightly higher due to discussions
    "youtube.com": 0.5,
    "tiktok.com": 0.3,
    "threads.net": 0.4,
}

# Known unreliable sources - very low credibility
UNRELIABLE_SOURCES = {
    # Add known fake news sites here
}


def extract_domain(url: str) -> str:
    """Extract the main domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_source_weight(url: str) -> float:
    """
    Get credibility weight for a source URL.
    
    Returns:
        float: Weight between 0.3 and 1.5
            - 1.5: Highly trusted (wire services, fact-checkers)
            - 1.3-1.4: Trusted news sources
            - 1.0: Default/unknown
            - 0.4-0.6: Social media
            - 0.3: Known unreliable
    """
    domain = extract_domain(url)
    
    if not domain:
        return 1.0
    
    # Check trusted sources first
    if domain in TRUSTED_SOURCES:
        return TRUSTED_SOURCES[domain]
    
    # Check for .edu or .gov domains
    if domain.endswith(".edu"):
        return TRUSTED_SOURCES.get("edu", 1.3)
    if domain.endswith(".gov") or domain.endswith(".gov.in"):
        return TRUSTED_SOURCES.get("gov", 1.4)
    
    # Check social media
    if domain in SOCIAL_MEDIA_SOURCES:
        return SOCIAL_MEDIA_SOURCES[domain]
    
    # Check unreliable sources
    if domain in UNRELIABLE_SOURCES:
        return UNRELIABLE_SOURCES[domain]
    
    # Default weight for unknown sources
    return 1.0


def is_social_media(url: str) -> bool:
    """Check if URL is from a social media platform."""
    domain = extract_domain(url)
    return domain in SOCIAL_MEDIA_SOURCES


def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted source."""
    domain = extract_domain(url)
    return domain in TRUSTED_SOURCES or domain.endswith(".edu") or domain.endswith(".gov")
