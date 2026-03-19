"""
Pytest configuration and fixtures for VeriFact tests.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_claim():
    """Sample claim for testing."""
    return "Python is a programming language"


@pytest.fixture
def sample_evidence_supporting():
    """Sample supporting evidence."""
    return {
        "url": "https://python.org",
        "best_sentence": "Python is a popular programming language.",
        "similarity": 0.92,
        "stance": "supports",
        "stance_score": 0.95,
        "source_weight": 1.0,
        "is_social_media": False
    }


@pytest.fixture
def sample_evidence_refuting():
    """Sample refuting evidence."""
    return {
        "url": "https://example.com",
        "best_sentence": "Python is not a programming language, it is a snake.",
        "similarity": 0.75,
        "stance": "refutes",
        "stance_score": 0.88,
        "source_weight": 1.0,
        "is_social_media": False
    }


@pytest.fixture
def sample_evidence_neutral():
    """Sample neutral evidence."""
    return {
        "url": "https://news.com",
        "best_sentence": "Programming languages are used worldwide.",
        "similarity": 0.45,
        "stance": "discusses",
        "stance_score": 0.70,
        "source_weight": 1.0,
        "is_social_media": False
    }


@pytest.fixture
def sample_social_media_evidence():
    """Sample social media evidence with lower weight."""
    return {
        "url": "https://twitter.com/user/status/123",
        "best_sentence": "Python is definitely a programming language!",
        "similarity": 0.88,
        "stance": "supports",
        "stance_score": 0.90,
        "source_weight": 0.5,
        "is_social_media": True
    }
