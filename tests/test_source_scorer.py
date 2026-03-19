"""
Unit tests for the Source Scorer module.
"""

import pytest
from app.core.source_scorer import (
    get_source_weight,
    is_social_media,
    is_trusted_source,
    extract_domain
)


class TestExtractDomain:
    """Tests for domain extraction."""
    
    def test_simple_url(self):
        """Extract domain from simple URL."""
        assert extract_domain("https://example.com/page") == "example.com"
    
    def test_www_prefix(self):
        """Should remove www prefix."""
        assert extract_domain("https://www.example.com") == "example.com"
    
    def test_subdomain(self):
        """Should preserve subdomains."""
        assert extract_domain("https://news.bbc.co.uk/article") == "news.bbc.co.uk"
    
    def test_invalid_url(self):
        """Invalid URL should return empty string."""
        assert extract_domain("not a url") == ""


class TestGetSourceWeight:
    """Tests for source weight calculation."""
    
    def test_trusted_sources(self):
        """Trusted sources should have weight > 1.0."""
        assert get_source_weight("https://reuters.com/article") == 1.5
        assert get_source_weight("https://bbc.com/news") == 1.4
        assert get_source_weight("https://snopes.com/fact-check") == 1.5
    
    def test_social_media_sources(self):
        """Social media should have weight < 1.0."""
        assert get_source_weight("https://twitter.com/user/status") == 0.5
        assert get_source_weight("https://facebook.com/post") == 0.4
        assert get_source_weight("https://reddit.com/r/news") == 0.6
    
    def test_unknown_sources(self):
        """Unknown sources should have default weight 1.0."""
        assert get_source_weight("https://randomsite.com") == 1.0
    
    def test_edu_domains(self):
        """Educational domains should be trusted."""
        assert get_source_weight("https://stanford.edu/research") == 1.3
    
    def test_gov_domains(self):
        """Government domains should be trusted."""
        assert get_source_weight("https://cdc.gov/info") == 1.4


class TestIsSocialMedia:
    """Tests for social media detection."""
    
    def test_twitter(self):
        assert is_social_media("https://twitter.com/user") == True
    
    def test_x_domain(self):
        assert is_social_media("https://x.com/user") == True
    
    def test_facebook(self):
        assert is_social_media("https://facebook.com/page") == True
    
    def test_reddit(self):
        assert is_social_media("https://reddit.com/r/news") == True
    
    def test_news_site(self):
        assert is_social_media("https://bbc.com/news") == False


class TestIsTrustedSource:
    """Tests for trusted source detection."""
    
    def test_reuters(self):
        assert is_trusted_source("https://reuters.com/article") == True
    
    def test_fact_checker(self):
        assert is_trusted_source("https://snopes.com/fact-check") == True
    
    def test_edu_domain(self):
        assert is_trusted_source("https://mit.edu/research") == True
    
    def test_random_site(self):
        assert is_trusted_source("https://randomsite.com") == False
    
    def test_social_media_not_trusted(self):
        assert is_trusted_source("https://twitter.com/user") == False
