"""
Unit tests for the Query Generator module.
"""

import pytest
from app.core.query_generator import generate_queries


class TestGenerateQueries:
    """Tests for query generation."""
    
    def test_basic_query_generation(self, sample_claim):
        """Should generate multiple queries from a claim."""
        queries = generate_queries(sample_claim)
        assert len(queries) >= 5
        assert isinstance(queries, list)
    
    def test_contains_original_claim(self, sample_claim):
        """Should include the original claim as a query."""
        queries = generate_queries(sample_claim)
        assert sample_claim.lower() in queries
    
    def test_contains_fact_check_query(self, sample_claim):
        """Should include fact check variation."""
        queries = generate_queries(sample_claim)
        assert any("fact check" in q for q in queries)
    
    def test_contains_hoax_query(self, sample_claim):
        """Should include hoax variation."""
        queries = generate_queries(sample_claim)
        assert any("hoax" in q for q in queries)
    
    def test_no_duplicates(self, sample_claim):
        """Should not contain duplicate queries."""
        queries = generate_queries(sample_claim)
        assert len(queries) == len(set(queries))
    
    def test_entity_based_queries(self):
        """Should generate entity-based queries when entities present."""
        claim = "Elon Musk announced Tesla layoffs"
        queries = generate_queries(claim)
        
        # Should have queries with entity names
        has_entity_query = any("Elon Musk" in q or "Tesla" in q for q in queries)
        assert has_entity_query
    
    def test_empty_claim(self):
        """Empty claim should return basic queries."""
        queries = generate_queries("")
        assert len(queries) >= 1  # At least the base query variations
