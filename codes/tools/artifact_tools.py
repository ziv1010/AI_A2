"""
Artifact Management Tools for the Agentic AI Framework.
Provides tools for saving, loading, and managing ML artifacts.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
from langchain_core.tools import tool
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ARTIFACTS_DIR


@tool
def list_artifacts() -> str:
    """
    List all artifacts saved in the artifacts directory.
    Shows file names, types, sizes, and creation times.

    Returns:
        Formatted string listing all artifacts.
    """
    try:
        artifacts = list(ARTIFACTS_DIR.glob("*"))
        if not artifacts:
            return "No artifacts found. Use save_artifact to create some."

        result = "=== Saved Artifacts ===\n"
        for f in sorted(artifacts, key=lambda x: x.stat().st_mtime, reverse=True):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                result += f"  {f.name} ({size_kb:.1f} KB) - {mtime.strftime('%Y-%m-%d %H:%M')}\n"

        return result
    except Exception as e:
        return f"Error listing artifacts: {str(e)}"


@tool
def read_artifact_text(filename: str) -> str:
    """
    Read a text-based artifact file (txt, json, csv, md).

    Args:
        filename: Name of the artifact file to read.

    Returns:
        Contents of the file as string.
    """
    try:
        filepath = ARTIFACTS_DIR / filename
        if not filepath.exists():
            return f"Error: Artifact '{filename}' not found."

        # Check if it's a text-based file
        text_extensions = {'.txt', '.json', '.csv', '.md', '.log', '.yaml', '.yml'}
        if filepath.suffix.lower() not in text_extensions:
            return f"Error: Cannot read binary file '{filename}'. Use load_artifact for pickle/joblib files."

        with open(filepath, 'r') as f:
            content = f.read()

        # Truncate if too long
        if len(content) > 8000:
            content = content[:8000] + "\n... [TRUNCATED]"

        return f"=== {filename} ===\n{content}"
    except Exception as e:
        return f"Error reading artifact: {str(e)}"


@tool
def save_artifact(name: str, content: str, artifact_type: str = "text") -> str:
    """
    Save an artifact to the artifacts directory.

    Args:
        name: Name for the artifact (without extension)
        content: Content to save (string for text, or JSON string for structured data)
        artifact_type: Type of artifact - 'text', 'json', or 'report'

    Returns:
        Confirmation message with file path.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if artifact_type == "json":
            filename = f"{name}_{timestamp}.json"
            filepath = ARTIFACTS_DIR / filename
            # Parse and re-dump to ensure valid JSON
            data = json.loads(content) if isinstance(content, str) else content
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        elif artifact_type == "report":
            filename = f"{name}_{timestamp}.md"
            filepath = ARTIFACTS_DIR / filename
            with open(filepath, 'w') as f:
                f.write(content)

        else:  # text
            filename = f"{name}_{timestamp}.txt"
            filepath = ARTIFACTS_DIR / filename
            with open(filepath, 'w') as f:
                f.write(content)

        return f"Artifact saved: {filepath}"
    except Exception as e:
        return f"Error saving artifact: {str(e)}"


@tool
def load_artifact(filename: str) -> str:
    """
    Load a pickled/joblib artifact (like trained models).

    Args:
        filename: Name of the artifact file.

    Returns:
        Status message and type of loaded object.
    """
    try:
        filepath = ARTIFACTS_DIR / filename
        if not filepath.exists():
            return f"Error: Artifact '{filename}' not found."

        if filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
        elif filepath.suffix == '.joblib':
            import joblib
            obj = joblib.load(filepath)
        else:
            return f"Error: Unknown format '{filepath.suffix}'. Use read_artifact_text for text files."

        return f"Loaded {type(obj).__name__} from {filename}"
    except Exception as e:
        return f"Error loading artifact: {str(e)}"


# Non-tool helper functions for direct Python use
def save_model(model: Any, name: str) -> Path:
    """Save a trained model using joblib (for internal use)."""
    import joblib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = ARTIFACTS_DIR / f"{name}_{timestamp}.joblib"
    joblib.dump(model, filepath)
    return filepath


def load_model(filename: str) -> Any:
    """Load a trained model (for internal use)."""
    import joblib
    filepath = ARTIFACTS_DIR / filename
    return joblib.load(filepath)


def save_results(results: Dict, name: str) -> Path:
    """Save results dictionary as JSON (for internal use)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = ARTIFACTS_DIR / f"{name}_{timestamp}.json"
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return filepath


def get_latest_artifact(pattern: str) -> Optional[Path]:
    """Get the most recently created artifact matching a pattern."""
    matches = list(ARTIFACTS_DIR.glob(pattern))
    if not matches:
        return None
    return max(matches, key=lambda x: x.stat().st_mtime)
