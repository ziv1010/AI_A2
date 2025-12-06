"""
Tools module for the Agentic AI Framework.
Provides data operations, code execution, artifact management, and prediction tools.
"""

from .data_tools import (
    list_data_files,
    load_dataframe_head,
    get_dataframe_info,
    get_column_statistics,
)

from .code_executor import (
    execute_code,
    run_python,
    run_code,
    reset_persistent_env,
    get_persistent_env,
)

from .artifact_tools import (
    save_artifact,
    load_artifact,
    list_artifacts,
    read_artifact_text,
)

from .prediction_tools import (
    predict_single_customer,
    get_customer_risk_profile,
    get_high_risk_customers,
    analyze_churn_factors,
)

__all__ = [
    # Data tools
    "list_data_files",
    "load_dataframe_head",
    "get_dataframe_info",
    "get_column_statistics",
    # Code execution
    "execute_code",
    "run_python",
    "run_code",
    "reset_persistent_env",
    "get_persistent_env",
    # Artifact management
    "save_artifact",
    "load_artifact",
    "list_artifacts",
    "read_artifact_text",
    # Prediction tools
    "predict_single_customer",
    "get_customer_risk_profile",
    "get_high_risk_customers",
    "analyze_churn_factors",
]
