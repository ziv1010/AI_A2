"""
Prediction Tools for the Agentic AI Framework.
Provides tools for making predictions on customer data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, ARTIFACTS_DIR


def load_model_and_prepare():
    """Load the trained model and prepare encoders."""
    import joblib
    from sklearn.preprocessing import LabelEncoder

    model_path = ARTIFACTS_DIR / "best_model.joblib"
    if not model_path.exists():
        return None, None, "Model not found. Please train a model first."

    model = joblib.load(model_path)

    # Load data to get encoders
    df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

    # Create encoders for categorical columns
    encoders = {}
    categorical_cols = ['Geography', 'Gender', 'Card Type']
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

    return model, encoders, None


@tool
def predict_single_customer(
    credit_score: int,
    geography: str,
    gender: str,
    age: int,
    tenure: int,
    balance: float,
    num_of_products: int,
    has_cr_card: int,
    is_active_member: int,
    estimated_salary: float,
    complain: int,
    satisfaction_score: int,
    card_type: str,
    point_earned: int
) -> str:
    """
    Predict churn probability for a single customer.

    Args:
        credit_score: Customer's credit score (300-850)
        geography: Country (France, Spain, Germany)
        gender: Gender (Male, Female)
        age: Customer's age
        tenure: Years as customer (0-10)
        balance: Account balance
        num_of_products: Number of products (1-4)
        has_cr_card: Has credit card (0 or 1)
        is_active_member: Is active member (0 or 1)
        estimated_salary: Estimated annual salary
        complain: Has complained (0 or 1)
        satisfaction_score: Satisfaction score (1-5)
        card_type: Card type (DIAMOND, GOLD, SILVER, PLATINUM)
        point_earned: Loyalty points earned

    Returns:
        Prediction result with churn probability and risk level.
    """
    try:
        model, encoders, error = load_model_and_prepare()
        if error:
            return error

        # Create feature dataframe
        data = {
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary],
            'Complain': [complain],
            'Satisfaction Score': [satisfaction_score],
            'Card Type': [card_type],
            'Point Earned': [point_earned]
        }

        df = pd.DataFrame(data)

        # Encode categorical variables
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        # Make prediction
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(df)[0][1]
            prediction = 1 if prob >= 0.5 else 0
        else:
            prediction = model.predict(df)[0]
            prob = float(prediction)

        # Determine risk level
        if prob >= 0.7:
            risk = "HIGH RISK"
        elif prob >= 0.4:
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"

        result = f"""
=== CHURN PREDICTION ===
Churn Probability: {prob*100:.1f}%
Risk Level: {risk}
Prediction: {'Will Churn' if prediction == 1 else 'Will Stay'}

Recommendation: {'Immediate retention action needed!' if risk == 'HIGH RISK' else 'Monitor customer engagement' if risk == 'MEDIUM RISK' else 'Maintain current relationship'}
"""
        return result

    except Exception as e:
        return f"Prediction error: {str(e)}"


@tool
def get_customer_risk_profile(customer_id: int) -> str:
    """
    Get risk profile for an existing customer by their ID.

    Args:
        customer_id: The customer's ID from the dataset

    Returns:
        Customer's risk profile and churn prediction.
    """
    try:
        model, encoders, error = load_model_and_prepare()
        if error:
            return error

        # Load data
        df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

        # Find customer
        customer = df[df['CustomerId'] == customer_id]
        if customer.empty:
            return f"Customer ID {customer_id} not found in dataset."

        customer = customer.iloc[0]

        # Prepare features
        feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                       'EstimatedSalary', 'Complain', 'Satisfaction Score',
                       'Card Type', 'Point Earned']

        X = df[df['CustomerId'] == customer_id][feature_cols].copy()

        # Encode
        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])

        # Predict
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = float(model.predict(X)[0])

        if prob >= 0.7:
            risk = "HIGH RISK"
        elif prob >= 0.4:
            risk = "MEDIUM RISK"
        else:
            risk = "LOW RISK"

        actual = "Churned" if customer['Exited'] == 1 else "Active"

        result = f"""
=== CUSTOMER PROFILE: {customer_id} ===
Name: {customer['Surname']}
Geography: {customer['Geography']}
Age: {customer['Age']}
Tenure: {customer['Tenure']} years
Balance: ${customer['Balance']:,.2f}
Products: {customer['NumOfProducts']}
Active Member: {'Yes' if customer['IsActiveMember'] == 1 else 'No'}
Has Complained: {'Yes' if customer['Complain'] == 1 else 'No'}
Satisfaction: {customer['Satisfaction Score']}/5

=== RISK ASSESSMENT ===
Churn Probability: {prob*100:.1f}%
Risk Level: {risk}
Actual Status: {actual}
"""
        return result

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_high_risk_customers(top_n: Optional[int] = None) -> str:
    """
    Get list of highest risk customers who are likely to churn.

    Args:
        top_n: Number of top risk customers to return. If not provided, defaults to 10.

    Returns:
        List of high-risk customers with their details.
    """
    # Handle None or invalid values
    if top_n is None or not isinstance(top_n, int) or top_n <= 0:
        top_n = 10
    try:
        model, encoders, error = load_model_and_prepare()
        if error:
            return error

        # Load data
        df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

        # Prepare features
        feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                       'EstimatedSalary', 'Complain', 'Satisfaction Score',
                       'Card Type', 'Point Earned']

        X = df[feature_cols].copy()

        # Encode
        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)

        df['ChurnProb'] = probs

        # Get active customers only (not already churned)
        active = df[df['Exited'] == 0].nlargest(top_n, 'ChurnProb')

        result = f"=== TOP {top_n} HIGH RISK CUSTOMERS (Most Likely to Churn) ===\n\n"
        for i, (_, row) in enumerate(active.iterrows(), 1):
            result += f"{i}. Customer ID: {row['CustomerId']} | Name: {row['Surname']}\n"
            result += f"   Churn Probability: {row['ChurnProb']*100:.1f}%\n"
            result += f"   Age: {row['Age']} | Balance: ${row['Balance']:,.0f} | Products: {row['NumOfProducts']}\n"
            result += f"   Active Member: {'Yes' if row['IsActiveMember'] else 'No'} | Has Complained: {'Yes' if row['Complain'] else 'No'}\n"
            result += f"   Geography: {row['Geography']} | Tenure: {row['Tenure']} years | Credit Score: {row['CreditScore']}\n\n"

        return result

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_low_risk_customers(top_n: Optional[int] = None) -> str:
    """
    Get list of lowest risk customers who are least likely to churn.

    Args:
        top_n: Number of low risk customers to return. If not provided, defaults to 10.

    Returns:
        List of low-risk customers with their details.
    """
    # Handle None or invalid values
    if top_n is None or not isinstance(top_n, int) or top_n <= 0:
        top_n = 10
    try:
        model, encoders, error = load_model_and_prepare()
        if error:
            return error

        # Load data
        df = pd.read_csv(DATA_DIR / "Customer-Churn-Records.csv")

        # Prepare features
        feature_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                       'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                       'EstimatedSalary', 'Complain', 'Satisfaction Score',
                       'Card Type', 'Point Earned']

        X = df[feature_cols].copy()

        # Encode
        for col, encoder in encoders.items():
            if col in X.columns:
                X[col] = encoder.transform(X[col])

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X)

        df['ChurnProb'] = probs

        # Get active customers only (not already churned), lowest risk
        active = df[df['Exited'] == 0].nsmallest(top_n, 'ChurnProb')

        result = f"=== TOP {top_n} LOW RISK CUSTOMERS (Least Likely to Churn) ===\n\n"
        for i, (_, row) in enumerate(active.iterrows(), 1):
            result += f"{i}. Customer ID: {row['CustomerId']} | Name: {row['Surname']}\n"
            result += f"   Churn Probability: {row['ChurnProb']*100:.1f}%\n"
            result += f"   Age: {row['Age']} | Balance: ${row['Balance']:,.0f} | Products: {row['NumOfProducts']}\n"
            result += f"   Active Member: {'Yes' if row['IsActiveMember'] else 'No'} | Has Complained: {'Yes' if row['Complain'] else 'No'}\n"
            result += f"   Geography: {row['Geography']} | Tenure: {row['Tenure']} years | Credit Score: {row['CreditScore']}\n\n"

        return result

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def analyze_churn_factors() -> str:
    """
    Analyze the main factors contributing to customer churn.
    Uses the trained model's feature importance.

    Returns:
        Analysis of top churn factors with recommendations.
    """
    try:
        model, _, error = load_model_and_prepare()
        if error:
            return error

        feature_names = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                        'EstimatedSalary', 'Complain', 'Satisfaction Score',
                        'Card Type', 'Point Earned']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            result = "=== CHURN FACTOR ANALYSIS ===\n\n"
            result += "Top factors influencing churn:\n\n"

            for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
                result += f"{i}. {row['Feature']}: {row['Importance']*100:.1f}%\n"

            result += "\n=== RECOMMENDATIONS ===\n"
            top_feature = importance_df.iloc[0]['Feature']

            recommendations = {
                'Age': "Focus on younger customer engagement programs",
                'Balance': "Review pricing for low-balance customers",
                'NumOfProducts': "Cross-sell to single-product customers",
                'IsActiveMember': "Re-engage inactive members with incentives",
                'Complain': "Improve complaint resolution process",
                'Satisfaction Score': "Address low satisfaction scores proactively",
                'Geography': "Review regional service quality",
                'Tenure': "Strengthen new customer onboarding"
            }

            result += f"\nBased on top factor ({top_feature}):\n"
            result += recommendations.get(top_feature, "Review and improve customer experience")

            return result
        else:
            return "Feature importance not available for this model type."

    except Exception as e:
        return f"Error: {str(e)}"
