"""
Code Guardrail for the Agentic AI Framework.
Validates generated Python code before execution.
"""

import re
import ast
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass


@dataclass
class CodeValidationResult:
    """Result of code validation."""
    is_valid: bool
    issues: List[str]
    risk_level: str  # 'safe', 'caution', 'dangerous'
    blocked_operations: List[str]


class CodeGuardrail:
    """
    Code validation guardrail for safe execution.

    Checks:
    1. Syntax validation
    2. Dangerous function detection
    3. Import validation
    4. Resource access patterns
    5. Infinite loop detection
    """

    # Absolutely forbidden - will block execution
    BLOCKED_FUNCTIONS = {
        "eval", "exec", "compile", "__import__",
        "open",  # We'll allow this with restrictions
        "input", "breakpoint",
    }

    # Blocked modules
    BLOCKED_MODULES = {
        "os", "subprocess", "sys", "shutil",
        "socket", "http", "urllib", "requests",
        "ftplib", "smtplib", "telnetlib",
        "ctypes", "multiprocessing",
    }

    # Allowed modules for ML work
    ALLOWED_MODULES = {
        "pandas", "numpy", "sklearn", "scipy",
        "matplotlib", "seaborn", "plotly",
        "json", "pickle", "joblib",
        "warnings", "pathlib", "datetime",
        "collections", "itertools", "functools",
        "math", "statistics", "random",
    }

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r"while\s+True\s*:",  # Potential infinite loop
        r"for\s+\w+\s+in\s+iter\s*\(",  # Potential infinite iterator
        r"recursion",  # Check for recursive calls
        r"\.\.\/",  # Path traversal
        r"\/etc\/",  # System file access
        r"\/root\/",
        r"rm\s+-",  # Shell commands
        r"\|\s*sh",
        r";\s*sh",
    ]

    def __init__(self, allowed_paths: List[str] = None):
        self.allowed_paths = allowed_paths or []

    def validate(self, code: str) -> CodeValidationResult:
        """
        Validate Python code for safety.

        Args:
            code: Python code string to validate.

        Returns:
            CodeValidationResult with validation status and details.
        """
        issues = []
        blocked = []
        risk_level = "safe"

        # Check 1: Syntax validation
        syntax_valid, syntax_error = self._check_syntax(code)
        if not syntax_valid:
            issues.append(f"Syntax error: {syntax_error}")
            return CodeValidationResult(
                is_valid=False,
                issues=issues,
                risk_level="dangerous",
                blocked_operations=["syntax_error"]
            )

        # Check 2: AST-based analysis
        try:
            tree = ast.parse(code)
            ast_issues, ast_blocked = self._analyze_ast(tree)
            issues.extend(ast_issues)
            blocked.extend(ast_blocked)
        except Exception as e:
            issues.append(f"AST analysis failed: {str(e)}")

        # Check 3: Pattern-based checks
        pattern_issues = self._check_dangerous_patterns(code)
        if pattern_issues:
            issues.extend(pattern_issues)
            risk_level = "caution"

        # Check 4: Import analysis
        import_issues, import_blocked = self._analyze_imports(code)
        issues.extend(import_issues)
        blocked.extend(import_blocked)

        # Determine final risk level
        if blocked:
            risk_level = "dangerous"
        elif len(issues) > 2:
            risk_level = "caution"

        is_valid = risk_level != "dangerous"

        return CodeValidationResult(
            is_valid=is_valid,
            issues=issues,
            risk_level=risk_level,
            blocked_operations=blocked
        )

    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

    def _analyze_ast(self, tree: ast.AST) -> Tuple[List[str], List[str]]:
        """Analyze AST for dangerous patterns."""
        issues = []
        blocked = []

        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                func_name = self._get_func_name(node.func)
                if func_name in self.BLOCKED_FUNCTIONS:
                    blocked.append(func_name)
                    issues.append(f"Blocked function: {func_name}")

            # Check attribute access (for module.function patterns)
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    module = node.value.id
                    if module in self.BLOCKED_MODULES:
                        blocked.append(f"{module}.{node.attr}")
                        issues.append(f"Blocked module access: {module}.{node.attr}")

        return issues, blocked

    def _get_func_name(self, node) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _check_dangerous_patterns(self, code: str) -> List[str]:
        """Check for dangerous code patterns."""
        issues = []
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Suspicious pattern detected: {pattern}")
        return issues

    def _analyze_imports(self, code: str) -> Tuple[List[str], List[str]]:
        """Analyze import statements."""
        issues = []
        blocked = []

        # Find all import statements
        import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
        matches = re.findall(import_pattern, code)

        for match in matches:
            module = match[0] or match[1]
            if module in self.BLOCKED_MODULES:
                blocked.append(f"import {module}")
                issues.append(f"Blocked import: {module}")
            elif module not in self.ALLOWED_MODULES and not module.startswith("sklearn"):
                issues.append(f"Unrecognized import: {module} (proceeding with caution)")

        return issues, blocked


def validate_code(code: str) -> Tuple[bool, List[str], str]:
    """
    Convenience function to validate code.

    Args:
        code: Python code to validate.

    Returns:
        Tuple of (is_valid, issues, risk_level)
    """
    guardrail = CodeGuardrail()
    result = guardrail.validate(code)
    return result.is_valid, result.issues, result.risk_level
