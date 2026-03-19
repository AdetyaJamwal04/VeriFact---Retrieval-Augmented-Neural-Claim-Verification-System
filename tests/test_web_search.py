"""
Unit tests for the Web Search module.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
from app.core.web_search import web_search, is_social_media

class TestIsSocialMedia:
    def test_social_domains(self):
        assert is_social_media("https://twitter.com/user/123")
        assert is_social_media("https://www.facebook.com/post")
        assert is_social_media("https://reddit.com/r/news")
        
    def test_news_domains(self):
        assert not is_social_media("https://cnn.com/article")
        assert not is_social_media("https://bbc.co.uk")

class TestWebSearch:
    """Tests for the fallback search logic."""
    
    @patch.dict(os.environ, {"TAVILY_API_KEY": "fake_key"})
    @patch('app.core.web_search._tavily_search')
    def test_tavily_priority(self, mock_tavily):
        """Should use Tavily if key is present."""
        mock_tavily.return_value = [{"href": "http://test.com", "title": "Test", "body": "Content"}]
        
        results = web_search(["query"])
        
        mock_tavily.assert_called_once()
        assert len(results) == 1
        assert results[0]['href'] == "http://test.com"

    @patch.dict(os.environ, {"TAVILY_API_KEY": ""})  # No Tavily key
    @patch.dict(os.environ, {"BRAVE_API_KEY": "fake_key"})
    @patch('app.core.web_search._brave_search')
    def test_brave_fallback(self, mock_brave):
        """Should fallback to Brave if Tavily key missing."""
        mock_brave.return_value = [{"href": "http://brave.com", "title": "Brave", "body": "Content"}]
        
        results = web_search(["query"])
        
        mock_brave.assert_called_once()
        assert results[0]['href'] == "http://brave.com"

    @patch.dict(os.environ, {"TAVILY_API_KEY": ""})
    @patch.dict(os.environ, {"BRAVE_API_KEY": ""})
    @patch('app.core.web_search._ddg_search')
    def test_ddg_fallback(self, mock_ddg):
        """Should fallback to DDG if no keys present."""
        mock_ddg.return_value = [{"href": "http://ddg.com", "title": "DDG", "body": "Content"}]
        
        results = web_search(["query"])
        
        mock_ddg.assert_called_once()
        assert results[0]['href'] == "http://ddg.com"
