"""
Modeler Agent for the Agentic AI Framework.
Responsible for data preprocessing, model training, evaluation, and selection.
"""

from typing import List
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.code_executor import execute_python_code
from tools.artifact_tools import save_artifact, list_artifacts, read_artifact_text


# Token-efficient prompt
MODELER_PROMPT = """You are a ML Modeler Agent. Your job is to build and evaluate predictive models.

CONTEXT:
Dataset: {dataset_name}
Target: {target_column}
Features: {feature_columns}

WORKFLOW:
1. Load and preprocess data using execute_python_code:
   - Handle missing values
   - Encode categorical variables (LabelEncoder or OneHotEncoder)
   - Scale numeric features if needed
   - Split into train/test (80/20)

2. Train AT LEAST 3 different models:
   - LogisticRegression
   - RandomForestClassifier
   - GradientBoostingClassifier or another algorithm

3. Evaluate each model:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC if applicable
   - Use classification_report

4. Select the best model based on F1-score or AUC

5. Save the best model using joblib:
   ```python
   joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
   ```

OUTPUT REQUIREMENTS:
Provide a summary with:
- Preprocessing steps applied
- Each model's metrics (table format)
- Best model name and why
- Feature importance (if available)

RULES:
- Always use train_test_split with random_state=42
- Print metrics using print() so they appear in output
- Handle errors gracefully
- Be token-efficient in responses

CODE TEMPLATE:
```python
# Load data
df = pd.read_csv(DATA_DIR / "{dataset_name}")

# Preprocessing
# ... your code ...

# Train models
models = {{
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}}

results = {{}}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {{
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }}
    print(f"{{name}}: {{results[name]}}")
```"""


def get_modeler_tools() -> List[BaseTool]:
    """Get the tools available to the modeler agent."""
    return [
        execute_python_code,
        save_artifact,
        list_artifacts,
        read_artifact_text,
    ]


def create_modeler_agent(llm):
    """
    Create a modeler agent using LangGraph's ReAct pattern.

    Args:
        llm: The language model to use.

    Returns:
        A compiled LangGraph agent.
    """
    tools = get_modeler_tools()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=MODELER_PROMPT,
    )

    return agent


def get_modeler_prompt_template() -> ChatPromptTemplate:
    """Get a prompt template for the modeler agent."""
    return ChatPromptTemplate.from_messages([
        ("system", MODELER_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])


# Helper function to format the prompt with context
def format_modeler_prompt(dataset_name: str, target_column: str, feature_columns: List[str]) -> str:
    """Format the modeler prompt with dataset context."""
    return MODELER_PROMPT.format(
        dataset_name=dataset_name,
        target_column=target_column,
        feature_columns=", ".join(feature_columns) if feature_columns else "To be determined"
    )
