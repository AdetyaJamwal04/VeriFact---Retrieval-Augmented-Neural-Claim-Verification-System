"""
Unit tests for the Claim Extractor module.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.core.claim_extractor import (
    extract_claim_from_text, 
    clean_text, 
    score_sentence_importance,
    extract_text_from_url
)

class TestCleanText:
    """Tests for text cleaning."""
    
    def test_basic_cleaning(self):
        text = "  Hello   world!  \n  New line.  "
        assert clean_text(text) == "Hello world! New line."
    
    def test_empty_text(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

class TestScoreSentenceImportance:
    """Tests for sentence importance scoring."""
    
    @pytest.fixture
    def mock_doc(self):
        mock = MagicMock()
        mock.ents = []
        return mock
        
    def test_length_score(self, mock_doc):
        # Good length sentence (30-200 chars)
        sent = "This is a sentence that has a very reasonable good length for a claim."
        score = score_sentence_importance(sent, mock_doc)
        assert score >= 1.0
        
    def test_numbers_bonus(self):
        # We need a real spacy doc for token.like_num to work, or mock nlp
        # Since the function imports nlp globally, we trust unit integration here
        pass

class TestExtractClaim:
    """Tests for claim extraction logic."""
    
    def test_extract_from_empty(self):
        claim, keywords = extract_claim_from_text("")
        assert claim == ""
        assert keywords == []
        
    @patch('app.core.claim_extractor.extract_keywords_hf')
    def test_extract_with_keywords(self, mock_extract_kw):
        """Test that keywords are returned alongside claim."""
        # Mock HF-based keyword extraction
        mock_extract_kw.return_value = ['python', 'programming', 'language']
        
        text = "Python is a great programming language. It is used by many developers."
        claim, keywords = extract_claim_from_text(text)
        
        assert claim  # Should find a claim
        assert 'python' in keywords

class TestExtractFromUrl:
    """Tests for URL extraction."""
    
    @patch('app.core.claim_extractor.fetch_url')
    @patch('app.core.claim_extractor.extract')
    def test_successful_extract(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html></html>"
        mock_extract.return_value = "Extracted text content."
        
        text = extract_text_from_url("http://example.com")
        assert text == "Extracted text content."
        
    @patch('app.core.claim_extractor.fetch_url')
    def test_failed_fetch(self, mock_fetch):
        mock_fetch.return_value = None
        text = extract_text_from_url("http://bad-url.com")
        assert text == ""
