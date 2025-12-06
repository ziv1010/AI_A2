"""
Utility functions for the Agentic AI Framework.
Provides helpers for rate limiting, logging, and common operations.
"""

import time
import functools
import logging
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from pathlib import Path

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the framework.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("agentic_ai")
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Helps stay within Groq's free tier limits.
    """

    def __init__(
        self,
        requests_per_minute: int = 25,
        tokens_per_minute: int = 5000
    ):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_times: List[float] = []
        self.token_counts: List[tuple] = []  # (timestamp, token_count)

    def wait_if_needed(self, estimated_tokens: int = 500) -> float:
        """
        Wait if rate limits would be exceeded.

        Args:
            estimated_tokens: Estimated tokens for next request

        Returns:
            Time waited in seconds
        """
        current_time = time.time()
        waited = 0.0

        # Clean old entries (older than 60 seconds)
        self.request_times = [
            t for t in self.request_times
            if current_time - t < 60
        ]
        self.token_counts = [
            (t, c) for t, c in self.token_counts
            if current_time - t < 60
        ]

        # Check RPM
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time)
                waited += wait_time
                current_time = time.time()

        # Check TPM
        current_tokens = sum(c for _, c in self.token_counts)
        if current_tokens + estimated_tokens > self.tpm_limit:
            # Wait for oldest tokens to expire
            if self.token_counts:
                wait_time = 60 - (current_time - self.token_counts[0][0])
                if wait_time > 0:
                    time.sleep(wait_time)
                    waited += wait_time

        return waited

    def record_request(self, token_count: int = 500):
        """Record a completed request."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_counts.append((current_time, token_count))


def rate_limited(rpm: int = 25, tpm: int = 5000):
    """Decorator to apply rate limiting to functions."""
    limiter = RateLimiter(rpm, tpm)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_if_needed()
            result = func(*args, **kwargs)
            limiter.record_request()
            return result
        return wrapper
    return decorator


# =============================================================================
# RETRY LOGIC
# =============================================================================

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)

            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    Uses a simple heuristic (4 chars per token average).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    # Simple heuristic: ~4 characters per token
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens

    Returns:
        Truncated text
    """
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text

    # Truncate based on character ratio
    target_chars = max_tokens * 4
    return text[:target_chars] + "\n... [TRUNCATED]"


# =============================================================================
# MESSAGE HELPERS
# =============================================================================

def extract_final_response(messages: List[Any]) -> str:
    """
    Extract the final AI response from a list of messages.

    Args:
        messages: List of message objects

    Returns:
        Content of the last AI message
    """
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return ""


def format_messages_for_context(messages: List[Any], max_tokens: int = 2000) -> str:
    """
    Format messages into a context string, truncated to max tokens.

    Args:
        messages: List of message objects
        max_tokens: Maximum tokens for output

    Returns:
        Formatted context string
    """
    parts = []

    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        content = getattr(msg, "content", str(msg))
        parts.append(f"{role}: {content}")

    full_context = "\n\n".join(parts)
    return truncate_to_tokens(full_context, max_tokens)


# =============================================================================
# FILE HELPERS
# =============================================================================

def safe_read_file(filepath: Path, max_size_kb: int = 1000) -> Optional[str]:
    """
    Safely read a file with size limits.

    Args:
        filepath: Path to file
        max_size_kb: Maximum file size in KB

    Returns:
        File contents or None if too large/error
    """
    try:
        if not filepath.exists():
            return None

        size_kb = filepath.stat().st_size / 1024
        if size_kb > max_size_kb:
            return f"[File too large: {size_kb:.1f} KB > {max_size_kb} KB limit]"

        return filepath.read_text()
    except Exception as e:
        return f"[Error reading file: {e}]"


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists, creating if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def format_model_results(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format model results as a readable table.

    Args:
        results: Dictionary of model_name -> metrics

    Returns:
        Formatted table string
    """
    if not results:
        return "No results available."

    # Get all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())

    metrics = sorted(all_metrics)

    # Build header
    header = "| Model | " + " | ".join(m.capitalize() for m in metrics) + " |"
    separator = "|" + "|".join("-" * (len(h) + 2) for h in header.split("|")[1:-1]) + "|"

    # Build rows
    rows = []
    for model_name, model_metrics in results.items():
        values = [f"{model_metrics.get(m, 'N/A'):.4f}" if isinstance(model_metrics.get(m), float) else str(model_metrics.get(m, 'N/A')) for m in metrics]
        row = f"| {model_name} | " + " | ".join(values) + " |"
        rows.append(row)

    return "\n".join([header, separator] + rows)


def create_summary_report(
    profiler_output: str,
    modeler_output: str,
    action_output: str,
    guardrail_result: Any = None
) -> str:
    """
    Create a comprehensive summary report.

    Args:
        profiler_output: Output from profiler agent
        modeler_output: Output from modeler agent
        action_output: Output from action agent
        guardrail_result: Optional guardrail pipeline result

    Returns:
        Formatted report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
# Agentic AI Predictive Analytics Report
Generated: {timestamp}

---

## 1. Data Profiling Summary

{truncate_to_tokens(profiler_output, 500)}

---

## 2. Model Building Summary

{truncate_to_tokens(modeler_output, 500)}

---

## 3. Insights and Recommendations

{truncate_to_tokens(action_output, 500)}

---

## 4. Quality Assurance

"""

    if guardrail_result:
        report += f"""
- Pipeline Status: {'PASSED' if guardrail_result.passed else 'NEEDS REVIEW'}
- Safety Score: {guardrail_result.overall_score:.2f}
- Issues Found: {len(guardrail_result.all_issues)}
"""
        if guardrail_result.all_issues:
            report += "\nIssues:\n"
            for issue in guardrail_result.all_issues[:5]:
                report += f"  - {issue}\n"
    else:
        report += "Guardrails not run."

    report += """
---

*Report generated by Agentic AI Framework*
"""

    return report
