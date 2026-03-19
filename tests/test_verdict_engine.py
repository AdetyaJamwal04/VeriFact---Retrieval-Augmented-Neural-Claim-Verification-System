"""
Unit tests for the Verdict Engine module.
"""

import pytest
from app.core.verdict_engine import compute_weighted_score, compute_final_verdict, sigmoid


class TestSigmoid:
    """Tests for the sigmoid function."""
    
    def test_sigmoid_zero(self):
        """Sigmoid of 0 should be 0.5."""
        assert sigmoid(0) == 0.5
    
    def test_sigmoid_positive(self):
        """Sigmoid of positive number should be > 0.5."""
        assert sigmoid(2) > 0.5
        assert sigmoid(2) < 1.0
    
    def test_sigmoid_negative(self):
        """Sigmoid of negative number should be < 0.5."""
        assert sigmoid(-2) < 0.5
        assert sigmoid(-2) > 0.0
    
    def test_sigmoid_large_positive(self):
        """Sigmoid of large positive number should approach 1."""
        assert sigmoid(10) > 0.99


class TestComputeWeightedScore:
    """Tests for compute_weighted_score function."""
    
    def test_supporting_evidence(self, sample_evidence_supporting):
        """Supporting evidence should have positive score."""
        score, stance = compute_weighted_score(sample_evidence_supporting)
        assert score > 0
        assert stance == "supports"
        # With similarity_boost: 0.92 * 0.95 * 1 * 1.0 * (1 + (0.92-0.5)*0.5) = 1.058
        assert score > 0.8  # Score should be high for strong supporting evidence
    
    def test_refuting_evidence(self, sample_evidence_refuting):
        """Refuting evidence should have negative score."""
        score, stance = compute_weighted_score(sample_evidence_refuting)
        assert score < 0
        assert stance == "refutes"
    
    def test_neutral_evidence(self, sample_evidence_neutral):
        """Neutral evidence should have zero score."""
        score, stance = compute_weighted_score(sample_evidence_neutral)
        assert score == 0
        assert stance == "discusses"
    
    def test_social_media_lower_weight(self, sample_social_media_evidence):
        """Social media evidence should have lower weight."""
        score, stance = compute_weighted_score(sample_social_media_evidence)
        # With similarity_boost: 0.88 * 0.90 * 1 * 0.5 * (1 + (0.88-0.5)*0.5) = 0.471
        assert score > 0.3 and score < 0.6  # Social media should have moderate positive score


class TestComputeFinalVerdict:
    """Tests for compute_final_verdict function."""
    
    def test_likely_true_verdict(self, sample_evidence_supporting):
        """Strong supporting evidence should return LIKELY TRUE."""
        result = compute_final_verdict([sample_evidence_supporting])
        assert result["verdict"] == "LIKELY TRUE"
        assert result["confidence"] > 0.5
        assert result["net_score"] > 0.35  # Updated threshold
    
    def test_likely_false_verdict(self, sample_evidence_refuting):
        """Strong refuting evidence should return LIKELY FALSE."""
        result = compute_final_verdict([sample_evidence_refuting])
        assert result["verdict"] == "LIKELY FALSE"
        assert result["net_score"] < -0.35  # Updated threshold
    
    def test_unverified_with_no_evidence(self):
        """No evidence should return UNVERIFIED."""
        result = compute_final_verdict([])
        assert result["verdict"] == "UNVERIFIED"
        assert result["confidence"] == 0.0
        assert result["net_score"] == 0
    
    def test_mixed_verdict(self, sample_evidence_neutral):
        """Only neutral evidence should return MIXED."""
        result = compute_final_verdict([sample_evidence_neutral])
        assert result["verdict"] == "MIXED / MISLEADING"
    
    def test_conflicting_evidence(self, sample_evidence_supporting, sample_evidence_refuting):
        """Conflicting evidence should consider net score."""
        evidences = [sample_evidence_supporting, sample_evidence_refuting]
        result = compute_final_verdict(evidences)
        # Net score depends on which is stronger
        assert result["verdict"] in ["LIKELY TRUE", "LIKELY FALSE", "MIXED / MISLEADING"]
    
    def test_social_media_lower_impact(
        self, 
        sample_evidence_supporting, 
        sample_social_media_evidence
    ):
        """Social media should have lower impact on verdict."""
        # Regular evidence only
        result1 = compute_final_verdict([sample_evidence_supporting])
        
        # Social media only (same stance but lower weight)
        result2 = compute_final_verdict([sample_social_media_evidence])
        
        # Regular evidence should have higher net score
        assert result1["net_score"] > result2["net_score"]
