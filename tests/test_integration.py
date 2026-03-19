"""
Integration tests for the Flask Application.
"""

import pytest
from app_flask import app
from unittest.mock import patch

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestAPIIntegration:
    """End-to-end API tests with mocked internals."""
    
    def test_health_check(self, client):
        """Health check should return 200 and valid JSON."""
        rv = client.get('/api/health')
        assert rv.status_code == 200
        data = rv.get_json()
        assert data['status'] == 'healthy'
        assert 'uptime_seconds' in data
        
    def test_index_route(self, client):
        """Root route should serve index.html."""
        rv = client.get('/')
        assert rv.status_code == 200

    def test_check_claim_validation_error(self, client):
        """Should fail if no input provided."""
        rv = client.post('/api/check', json={})
        assert rv.status_code == 400
        assert 'error' in rv.get_json()

    @patch('app_flask.web_search')
    @patch('app_flask.build_evidence')
    @patch('app_flask.compute_final_verdict')
    def test_check_claim_success(self, mock_verdict, mock_evidence, mock_search, client):
        """Should process valid claim successfully."""
        # Mock pipeline responses
        mock_search.return_value = []
        mock_evidence.return_value = []
        mock_verdict.return_value = {
            "verdict": "UNVERIFIED",
            "confidence": 0.0,
            "net_score": 0.0,
            "explanation": "No evidence found"
        }
        
        payload = {
            "claim": "The earth is flat",
            "max_results": 3
        }
        
        rv = client.post('/api/check', json=payload)
        
        assert rv.status_code == 200
        data = rv.get_json()
        assert data['status'] == 'success'
        assert data['verdict'] == 'UNVERIFIED'
        
    def test_check_claim_invalid_url(self, client):
        """Should reject invalid URLs."""
        payload = {"url": "htt://bad-url", "max_results": 3}
        rv = client.post('/api/check', json=payload)
        assert rv.status_code == 400
