"""
Data Tools for the Agentic AI Framework.
Provides tools for data discovery, loading, and exploration.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR


@tool
def list_data_files() -> str:
    """
    List all available CSV data files in the data directory.
    Returns a formatted string with file names and sizes.
    Use this tool FIRST to discover what datasets are available.
    """
    try:
        csv_files = list(DATA_DIR.glob("*.csv"))
        if not csv_files:
            return "No CSV files found in data directory."

        result = "Available datasets:\n"
        for f in csv_files:
            size_kb = f.stat().st_size / 1024
            result += f"  - {f.name} ({size_kb:.1f} KB)\n"
        return result
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def load_dataframe_head(filename: str, n_rows: int = 5) -> str:
    """
    Load and display the first n rows of a CSV file.
    Use this to inspect the structure and sample data.

    Args:
        filename: Name of the CSV file (e.g., 'Customer-Churn-Records.csv')
        n_rows: Number of rows to display (default: 5, max: 10)

    Returns:
        String representation of the dataframe head with column info.
    """
    try:
        n_rows = min(n_rows, 10)  # Limit for token efficiency
        filepath = DATA_DIR / filename

        if not filepath.exists():
            return f"Error: File '{filename}' not found in data directory."

        df = pd.read_csv(filepath, nrows=n_rows)

        result = f"Dataset: {filename}\n"
        result += f"Columns ({len(df.columns)}): {', '.join(df.columns.tolist())}\n"
        result += f"\nFirst {n_rows} rows:\n"
        result += df.to_string(index=False, max_colwidth=30)

        return result
    except Exception as e:
        return f"Error loading file: {str(e)}"


@tool
def get_dataframe_info(filename: str) -> str:
    """
    Get comprehensive information about a dataset including:
    - Shape (rows, columns)
    - Column names and data types
    - Missing value counts
    - Memory usage

    Args:
        filename: Name of the CSV file

    Returns:
        Formatted string with dataset information.
    """
    try:
        filepath = DATA_DIR / filename

        if not filepath.exists():
            return f"Error: File '{filename}' not found."

        df = pd.read_csv(filepath)

        result = f"=== Dataset Info: {filename} ===\n"
        result += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        result += f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n\n"

        result += "Columns:\n"
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            unique = df[col].nunique()
            result += f"  {col}: {dtype} | {missing} missing | {unique} unique\n"

        # Identify potential target variable (binary columns)
        result += "\nPotential target columns (binary):\n"
        for col in df.columns:
            if df[col].nunique() == 2:
                result += f"  - {col}: {df[col].value_counts().to_dict()}\n"

        return result
    except Exception as e:
        return f"Error getting info: {str(e)}"


@tool
def get_column_statistics(filename: str, column: str) -> str:
    """
    Get detailed statistics for a specific column.

    Args:
        filename: Name of the CSV file
        column: Name of the column to analyze

    Returns:
        Statistical summary of the column.
    """
    try:
        filepath = DATA_DIR / filename
        df = pd.read_csv(filepath)

        if column not in df.columns:
            return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

        col_data = df[column]
        result = f"=== Statistics for '{column}' ===\n"

        if pd.api.types.is_numeric_dtype(col_data):
            result += f"Type: Numeric\n"
            result += f"Count: {col_data.count()}\n"
            result += f"Mean: {col_data.mean():.4f}\n"
            result += f"Std: {col_data.std():.4f}\n"
            result += f"Min: {col_data.min()}\n"
            result += f"25%: {col_data.quantile(0.25):.4f}\n"
            result += f"50%: {col_data.quantile(0.50):.4f}\n"
            result += f"75%: {col_data.quantile(0.75):.4f}\n"
            result += f"Max: {col_data.max()}\n"
        else:
            result += f"Type: Categorical\n"
            result += f"Unique values: {col_data.nunique()}\n"
            result += f"Top 10 values:\n"
            for val, count in col_data.value_counts().head(10).items():
                pct = count / len(col_data) * 100
                result += f"  {val}: {count} ({pct:.1f}%)\n"

        return result
    except Exception as e:
        return f"Error: {str(e)}"


# Non-tool helper functions for internal use
def load_full_dataframe(filename: str) -> pd.DataFrame:
    """Load complete dataframe (for internal use, not as tool)."""
    filepath = DATA_DIR / filename
    return pd.read_csv(filepath)


def get_target_column_candidates(filename: str) -> List[str]:
    """Identify potential target columns for classification."""
    df = load_full_dataframe(filename)
    candidates = []
    for col in df.columns:
        if df[col].nunique() == 2 or (df[col].nunique() <= 10 and df[col].dtype in ['int64', 'object']):
            candidates.append(col)
    return candidates
