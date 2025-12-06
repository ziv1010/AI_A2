"""
Output Guardrail for the Agentic AI Framework.
Validates LLM outputs before returning to user or executing.
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class OutputValidationResult:
    """Result of output validation."""
    is_valid: bool
    sanitized_output: str
    issues: List[str]
    confidence_score: float  # 0.0 to 1.0


class OutputGuardrail:
    """
    Multi-layer output validation guardrail.

    Checks:
    1. Sensitive information leakage
    2. Harmful content detection
    3. Format validation
    4. Consistency checks
    5. Confidence scoring
    """

    # Patterns for sensitive info that shouldn't be in output
    SENSITIVE_PATTERNS = [
        r"api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]",
        r"password\s*[:=]\s*['\"][^'\"]+['\"]",
        r"secret\s*[:=]\s*['\"][^'\"]+['\"]",
        r"token\s*[:=]\s*['\"][^'\"]+['\"]",
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI key pattern
        r"gsk_[a-zA-Z0-9]{20,}",  # Groq key pattern
    ]

    # Patterns indicating potential issues
    ISSUE_PATTERNS = [
        r"i\s+(don'?t|cannot|can'?t)\s+help\s+with",
        r"as\s+an\s+ai",
        r"i'?m\s+sorry,?\s+but",
        r"i\s+cannot\s+provide",
    ]

    # Uncertainty indicators
    UNCERTAINTY_PHRASES = [
        "i think",
        "i believe",
        "possibly",
        "might be",
        "could be",
        "not sure",
        "uncertain",
        "may or may not",
        "it's possible",
        "perhaps",
    ]

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def validate(self, output_text: str, context: Optional[Dict] = None) -> OutputValidationResult:
        """
        Validate output text through multiple checks.

        Args:
            output_text: The output to validate.
            context: Optional context for consistency checking.

        Returns:
            OutputValidationResult with validation status and details.
        """
        issues = []
        sanitized = output_text
        confidence = 1.0

        # Check 1: Sensitive information
        sensitive_matches = self._check_patterns(output_text, self.SENSITIVE_PATTERNS)
        if sensitive_matches:
            issues.append("Sensitive information detected and redacted")
            sanitized = self._redact_sensitive(sanitized)
            confidence -= 0.2

        # Check 2: AI refusal patterns (might indicate invalid request)
        refusal_matches = self._check_patterns(output_text, self.ISSUE_PATTERNS)
        if refusal_matches:
            issues.append("Output contains refusal/limitation statements")
            confidence -= 0.1

        # Check 3: Uncertainty analysis
        uncertainty_count = sum(
            1 for phrase in self.UNCERTAINTY_PHRASES
            if phrase.lower() in output_text.lower()
        )
        if uncertainty_count > 2:
            issues.append(f"High uncertainty detected ({uncertainty_count} indicators)")
            confidence -= 0.1 * min(uncertainty_count, 3)

        # Check 4: Empty or too short output
        if len(output_text.strip()) < 10:
            issues.append("Output is too short or empty")
            confidence -= 0.3

        # Check 5: Consistency with context (if provided)
        if context:
            consistency_issues = self._check_consistency(output_text, context)
            issues.extend(consistency_issues)
            confidence -= 0.1 * len(consistency_issues)

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        is_valid = confidence >= self.min_confidence and len(issues) <= 2

        return OutputValidationResult(
            is_valid=is_valid,
            sanitized_output=sanitized,
            issues=issues,
            confidence_score=confidence
        )

    def _check_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Check text against regex patterns."""
        matches = []
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern)
        return matches

    def _redact_sensitive(self, text: str) -> str:
        """Redact sensitive information from text."""
        for pattern in self.SENSITIVE_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text

    def _check_consistency(self, output: str, context: Dict) -> List[str]:
        """Check output consistency with provided context."""
        issues = []

        # Check if output mentions dataset that matches context
        if "dataset_name" in context:
            if context["dataset_name"] not in output and len(output) > 200:
                issues.append("Output may not reference the correct dataset")

        # Check if target column is mentioned when it should be
        if "target_column" in context:
            if context["target_column"] not in output and "target" in output.lower():
                issues.append("Target column name mismatch possible")

        return issues


def validate_output(output_text: str, context: Optional[Dict] = None) -> Tuple[bool, str, List[str], float]:
    """
    Convenience function to validate output.

    Args:
        output_text: Text to validate.
        context: Optional context dictionary.

    Returns:
        Tuple of (is_valid, sanitized_output, issues, confidence_score)
    """
    guardrail = OutputGuardrail()
    result = guardrail.validate(output_text, context)
    return result.is_valid, result.sanitized_output, result.issues, result.confidence_score
