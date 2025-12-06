"""
Code Executor Tool for the Agentic AI Framework.
Provides safe Python code execution capabilities with sandboxing.
"""

import sys
import io
import traceback
import contextlib
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.tools import tool
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, ARTIFACTS_DIR, EXECUTION_CONFIG


def create_safe_environment() -> Dict[str, Any]:
    """
    Create a safe execution environment with pre-imported libraries.
    Returns a dictionary to be used as globals() for exec().
    """
    safe_env = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
    }

    # Pre-import allowed libraries
    try:
        import pandas as pd
        import numpy as np
        safe_env["pd"] = pd
        safe_env["np"] = np
    except ImportError:
        pass

    try:
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        from sklearn.impute import SimpleImputer

        safe_env["train_test_split"] = train_test_split
        safe_env["cross_val_score"] = cross_val_score
        safe_env["StandardScaler"] = StandardScaler
        safe_env["LabelEncoder"] = LabelEncoder
        safe_env["RandomForestClassifier"] = RandomForestClassifier
        safe_env["GradientBoostingClassifier"] = GradientBoostingClassifier
        safe_env["LogisticRegression"] = LogisticRegression
        safe_env["DecisionTreeClassifier"] = DecisionTreeClassifier
        safe_env["accuracy_score"] = accuracy_score
        safe_env["precision_score"] = precision_score
        safe_env["recall_score"] = recall_score
        safe_env["f1_score"] = f1_score
        safe_env["confusion_matrix"] = confusion_matrix
        safe_env["classification_report"] = classification_report
        safe_env["roc_auc_score"] = roc_auc_score
        safe_env["SimpleImputer"] = SimpleImputer
    except ImportError:
        pass

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        safe_env["plt"] = plt
        safe_env["sns"] = sns
    except ImportError:
        pass

    try:
        import pickle
        import joblib
        import json
        safe_env["pickle"] = pickle
        safe_env["joblib"] = joblib
        safe_env["json"] = json
    except ImportError:
        pass

    # Add directory paths as strings for easier use
    safe_env["DATA_DIR"] = DATA_DIR
    safe_env["ARTIFACTS_DIR"] = ARTIFACTS_DIR
    safe_env["DATA_PATH"] = str(DATA_DIR)
    safe_env["ARTIFACTS_PATH"] = str(ARTIFACTS_DIR)
    safe_env["Path"] = Path

    # Suppress warnings
    safe_env["warnings"] = warnings

    return safe_env


# Global persistent environment
_PERSISTENT_ENV: Optional[Dict[str, Any]] = None


def get_persistent_env() -> Dict[str, Any]:
    """Get or create the persistent execution environment."""
    global _PERSISTENT_ENV
    if _PERSISTENT_ENV is None:
        _PERSISTENT_ENV = create_safe_environment()
    return _PERSISTENT_ENV


def reset_persistent_env():
    """Reset the persistent environment."""
    global _PERSISTENT_ENV
    _PERSISTENT_ENV = None


def run_code(code: str) -> str:
    """
    Execute Python code - internal function without @tool decorator.
    Use this directly when you need to run code programmatically.
    """
    try:
        env = get_persistent_env()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                exec(code, env)
            except Exception as e:
                error_tb = traceback.format_exc()
                return f"Execution Error:\n{error_tb}"

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        result = ""
        if stdout_output:
            result += stdout_output
        if stderr_output:
            result += f"\nWarnings/Errors:\n{stderr_output}"

        if not result.strip():
            result = "Code executed successfully (no output)."

        max_chars = EXECUTION_CONFIG.get("max_output_chars", 10000)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... [OUTPUT TRUNCATED]"

        return result

    except Exception as e:
        return f"System Error: {str(e)}"


@tool
def execute_code(code: str) -> str:
    """Execute Python code. Libraries available: pandas as pd, numpy as np, sklearn models and metrics, matplotlib as plt, seaborn as sns. Use DATA_DIR for data path. Use print() to see output.

    Args:
        code: Python code string to execute

    Returns:
        Output from code execution
    """
    return run_code(code)


@tool
def run_python(code: str) -> str:
    """Run Python code with pandas, numpy, sklearn pre-imported. DATA_DIR points to data folder. Always use print() for output.

    Args:
        code: The Python code to run

    Returns:
        The printed output or error message
    """
    return run_code(code)
