"""
Input Guardrail for the Agentic AI Framework.
Validates and sanitizes inputs before processing.
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class InputValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: str
    issues: List[str]
    risk_level: str  # 'low', 'medium', 'high'


class InputGuardrail:
    """
    Multi-layer input validation guardrail.

    Checks:
    1. Prompt injection detection
    2. Malicious code patterns
    3. PII detection
    4. Input length limits
    5. Encoding issues
    """

    # Patterns that might indicate prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions",
        r"disregard\s+(your|all)\s+instructions",
        r"you\s+are\s+now\s+",
        r"pretend\s+to\s+be",
        r"act\s+as\s+if",
        r"forget\s+(everything|your\s+instructions)",
        r"new\s+instructions:",
        r"system\s*:\s*",
        r"\[INST\]",
        r"<\|im_start\|>",
    ]

    # Dangerous code patterns
    DANGEROUS_CODE_PATTERNS = [
        r"os\.system\s*\(",
        r"subprocess\.",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__",
        r"open\s*\([^)]*['\"]\/etc",
        r"rm\s+-rf",
        r"shutil\.rmtree",
        r"\.env",
        r"api[_-]?key",
        r"password\s*=",
        r"secret\s*=",
    ]

    # PII patterns
    PII_PATTERNS = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",  # SSN
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{16}\b",  # Credit card
    ]

    def __init__(self, max_input_length: int = 10000):
        self.max_input_length = max_input_length

    def validate(self, input_text: str) -> InputValidationResult:
        """
        Validate input text through multiple checks.

        Args:
            input_text: The input to validate.

        Returns:
            InputValidationResult with validation status and details.
        """
        issues = []
        risk_level = "low"
        sanitized = input_text

        # Check 1: Length limit
        if len(input_text) > self.max_input_length:
            issues.append(f"Input exceeds max length ({self.max_input_length} chars)")
            sanitized = input_text[:self.max_input_length]
            risk_level = "medium"

        # Check 2: Prompt injection
        injection_matches = self._check_patterns(input_text, self.INJECTION_PATTERNS)
        if injection_matches:
            issues.append(f"Potential prompt injection detected: {injection_matches}")
            risk_level = "high"

        # Check 3: Dangerous code
        code_matches = self._check_patterns(input_text, self.DANGEROUS_CODE_PATTERNS)
        if code_matches:
            issues.append(f"Potentially dangerous code patterns: {code_matches}")
            risk_level = "high"

        # Check 4: PII detection (warning only)
        pii_matches = self._check_patterns(input_text, self.PII_PATTERNS)
        if pii_matches:
            issues.append(f"Potential PII detected (will be handled carefully)")
            if risk_level == "low":
                risk_level = "medium"

        # Check 5: Encoding issues
        try:
            sanitized.encode('utf-8').decode('utf-8')
        except UnicodeError:
            issues.append("Encoding issues detected, sanitizing")
            sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')

        is_valid = risk_level != "high"

        return InputValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            issues=issues,
            risk_level=risk_level
        )

    def _check_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Check text against a list of regex patterns."""
        matches = []
        text_lower = text.lower()
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(pattern)
        return matches


def validate_input(input_text: str) -> Tuple[bool, str, List[str]]:
    """
    Convenience function to validate input.

    Args:
        input_text: Text to validate.

    Returns:
        Tuple of (is_valid, sanitized_input, issues)
    """
    guardrail = InputGuardrail()
    result = guardrail.validate(input_text)
    return result.is_valid, result.sanitized_input, result.issues
