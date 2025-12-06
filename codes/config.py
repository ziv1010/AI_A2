"""
Configuration module for the Agentic AI Predictive Analytics Framework.
Manages API keys, model settings, and directory paths.
"""

import os
from pathlib import Path

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CODES_DIR = PROJECT_ROOT / "codes"
TOOLS_DIR = CODES_DIR / "tools"
AGENTS_DIR = CODES_DIR / "agents"
GUARDRAILS_DIR = CODES_DIR / "guardrails"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensure directories exist
for dir_path in [DATA_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API CONFIGURATION
# =============================================================================
# Set your GROQ API key here or via environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

def set_api_key(key: str):
    """Set the GROQ API key."""
    global GROQ_API_KEY
    GROQ_API_KEY = key
    os.environ["GROQ_API_KEY"] = key

# =============================================================================
# MODEL CONFIGURATION (Token-efficient settings for free tier)
# =============================================================================
MODEL_CONFIG = {
    "model_name": "llama-3.3-70b-versatile",  # Best free model on Groq
    "temperature": 0.1,  # Low temp for consistency
    "max_tokens": 1500,  # Conservative for free tier (6000 TPM limit)
    "timeout": 60,
}

# Alternative smaller model for less critical tasks
SMALL_MODEL_CONFIG = {
    "model_name": "llama-3.1-8b-instant",  # Faster, smaller model
    "temperature": 0.1,
    "max_tokens": 1000,
    "timeout": 30,
}

# =============================================================================
# RATE LIMITING (Groq Free Tier: 30 RPM, 6000 TPM)
# =============================================================================
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 25,  # Stay under 30 RPM
    "tokens_per_minute": 5000,  # Stay under 6000 TPM
    "retry_delay": 2.0,  # Seconds between retries
    "max_retries": 3,
}

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================
AGENT_CONFIG = {
    "max_iterations": 10,  # Max tool calls per agent
    "recursion_limit": 25,  # LangGraph recursion limit
    "checkpoint_enabled": True,
}

# =============================================================================
# CODE EXECUTION SAFETY
# =============================================================================
EXECUTION_CONFIG = {
    "timeout_seconds": 120,  # Max execution time
    "max_output_chars": 10000,  # Truncate long outputs
    "allowed_imports": [
        "pandas", "numpy", "sklearn", "scipy", "matplotlib",
        "seaborn", "json", "os", "pathlib", "warnings",
        "pickle", "joblib"
    ],
}

# =============================================================================
# GUARDRAIL CONFIGURATION
# =============================================================================
GUARDRAIL_CONFIG = {
    "enable_input_validation": True,
    "enable_output_validation": True,
    "enable_code_review": True,
    "enable_hallucination_check": True,
    "confidence_threshold": 0.7,
    "max_hallucination_score": 0.3,
}
