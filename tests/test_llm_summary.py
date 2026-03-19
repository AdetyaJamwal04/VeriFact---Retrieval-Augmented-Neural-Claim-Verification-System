"""
Unit tests for the LLM verdict summary feature.
Tests generate_verdict_summary() in llm_helper.py with mocked Groq calls.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestGenerateVerdictSummary:
    """Tests for the generate_verdict_summary function."""

    def setup_method(self):
        """Common test data."""
        self.claim = "The Earth is flat"
        self.evidence_list = [
            {
                "url": "https://nasa.gov/earth",
                "best_sentence": "NASA confirms Earth is an oblate spheroid.",
                "stance": "refutes",
                "stance_score": 0.95,
                "source_weight": 1.4,
                "is_social_media": False,
            },
            {
                "url": "https://bbc.com/science",
                "best_sentence": "Scientists have long established that Earth is round.",
                "stance": "refutes",
                "stance_score": 0.88,
                "source_weight": 1.4,
                "is_social_media": False,
            },
        ]
        self.explanation = {
            "decision_reason": "Score (-1.50) is below -0.35 threshold. The majority of credible evidence contradicts this claim.",
            "breakdown": {
                "support_count": 0,
                "refute_count": 2,
                "neutral_count": 0,
                "trusted_sources": 2,
                "total_sources": 2,
            },
        }

    @patch("app.core.llm_helper._call_groq")
    def test_returns_llm_summary_when_available(self, mock_groq):
        """When Groq is available, should return LLM-generated summary."""
        from app.core.llm_helper import generate_verdict_summary

        mock_groq.return_value = "Multiple trusted sources including NASA and BBC confirm that Earth is not flat but is an oblate spheroid. No credible evidence was found supporting this claim."

        result = generate_verdict_summary(
            claim=self.claim,
            verdict="LIKELY FALSE",
            confidence=0.85,
            evidence_list=self.evidence_list,
            explanation=self.explanation,
        )

        assert "NASA" in result or "BBC" in result or "spheroid" in result or "flat" in result
        assert len(result) > 20  # Should be a meaningful summary
        mock_groq.assert_called_once()

    @patch("app.core.llm_helper._call_groq")
    def test_falls_back_when_groq_unavailable(self, mock_groq):
        """When Groq returns None, should fall back to decision_reason."""
        from app.core.llm_helper import generate_verdict_summary

        mock_groq.return_value = None

        result = generate_verdict_summary(
            claim=self.claim,
            verdict="LIKELY FALSE",
            confidence=0.85,
            evidence_list=self.evidence_list,
            explanation=self.explanation,
        )

        assert result == self.explanation["decision_reason"]

    @patch("app.core.llm_helper._call_groq")
    def test_returns_fallback_with_no_evidence(self, mock_groq):
        """With no evidence, should return fallback without calling Groq."""
        from app.core.llm_helper import generate_verdict_summary

        result = generate_verdict_summary(
            claim=self.claim,
            verdict="UNVERIFIED",
            confidence=0.0,
            evidence_list=[],
            explanation=None,
        )

        assert "No relevant evidence" in result
        mock_groq.assert_not_called()

    @patch("app.core.llm_helper._call_groq")
    def test_strips_quotes_from_llm_response(self, mock_groq):
        """Should strip surrounding quotes from LLM response."""
        from app.core.llm_helper import generate_verdict_summary

        mock_groq.return_value = '"This is a quoted summary."'

        result = generate_verdict_summary(
            claim=self.claim,
            verdict="LIKELY FALSE",
            confidence=0.85,
            evidence_list=self.evidence_list,
            explanation=self.explanation,
        )

        assert not result.startswith('"')
        assert not result.endswith('"')
        assert result == "This is a quoted summary."

    @patch("app.core.llm_helper._call_groq")
    def test_fallback_without_explanation(self, mock_groq):
        """When Groq is unavailable and no explanation provided, uses generic fallback."""
        from app.core.llm_helper import generate_verdict_summary

        mock_groq.return_value = None

        result = generate_verdict_summary(
            claim=self.claim,
            verdict="LIKELY FALSE",
            confidence=0.85,
            evidence_list=self.evidence_list,
            explanation=None,
        )

        assert "LIKELY FALSE" in result
        assert "2 source" in result


class TestVerdictEngineWithSummary:
    """Test that compute_final_verdict includes summary in result."""

    def test_verdict_result_includes_summary_field(self):
        """compute_final_verdict should always include a 'summary' field."""
        from app.core.verdict_engine import compute_final_verdict

        evidences = [
            {
                "url": "https://example.com",
                "best_sentence": "Python is a programming language.",
                "similarity": 0.92,
                "stance": "supports",
                "stance_score": 0.95,
                "source_weight": 1.0,
                "is_social_media": False,
            }
        ]

        result = compute_final_verdict(evidences, claim="Python is a programming language")
        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_unverified_verdict_includes_summary(self):
        """UNVERIFIED verdict (no evidence) should also have a summary."""
        from app.core.verdict_engine import compute_final_verdict

        result = compute_final_verdict([], claim="Some claim")
        assert "summary" in result
        assert isinstance(result["summary"], str)
