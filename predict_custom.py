"""
╔══════════════════════════════════════════════════════════════════════╗
║  predict_custom.py — Predict creditworthiness for a new applicant   ║
║  Usage: python predict_custom.py                                     ║
║         python predict_custom.py --demo          (use preset values) ║
╚══════════════════════════════════════════════════════════════════════╝

Loads the saved model bundle (saved_model/best_credit_model.pkl) and
prompts the user for feature values, then returns:
  • CREDITWORTHY      (probability of default < 50 %)
  • NOT CREDITWORTHY  (probability of default ≥ 50 %)
"""

import argparse
import os
import sys
import joblib
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Feature metadata: (display_name, description, dtype, example_value)
# ─────────────────────────────────────────────────────────────────────
FEATURE_META = {
    "RevolvingUtilizationOfUnsecuredLines": (
        "Revolving Utilization of Unsecured Lines",
        "Total balance on credit cards & lines / credit limits  (e.g. 0.25 = 25 %)",
        float, 0.25,
    ),
    "age": (
        "Age",
        "Borrower's age in years",
        int, 45,
    ),
    "NumberOfTime30-59DaysPastDueNotWorse": (
        "# Times 30–59 Days Past Due (last 2 yrs)",
        "How many times 30–59 days late but no worse",
        int, 0,
    ),
    "DebtRatio": (
        "Debt Ratio",
        "Monthly debt payments / monthly gross income  (e.g. 0.35)",
        float, 0.35,
    ),
    "MonthlyIncome": (
        "Monthly Income (USD)",
        "Gross monthly income in USD",
        float, 5000.0,
    ),
    "NumberOfOpenCreditLinesAndLoans": (
        "# Open Credit Lines & Loans",
        "Number of open loans (installment + revolving)",
        int, 8,
    ),
    "NumberOfTimes90DaysLate": (
        "# Times 90+ Days Late (last 2 yrs)",
        "How many times 90+ days overdue",
        int, 0,
    ),
    "NumberRealEstateLoansOrLines": (
        "# Real Estate Loans or Lines",
        "Number of mortgage and real estate loans",
        int, 1,
    ),
    "NumberOfTime60-89DaysPastDueNotWorse": (
        "# Times 60–89 Days Past Due (last 2 yrs)",
        "How many times 60–89 days late but no worse",
        int, 0,
    ),
    "NumberOfDependents": (
        "# Dependents",
        "Number of dependents in family (spouse, children, etc.)",
        int, 2,
    ),
}


# ─────────────────────────────────────────────────────────────────────
# Helper: derive the engineered features (mirror credit_scoring.py)
# ─────────────────────────────────────────────────────────────────────
def derive_engineered(raw: dict) -> dict:
    """Compute the 5 engineered features from raw inputs."""
    income = raw["MonthlyIncome"]
    deps   = raw["NumberOfDependents"]

    engineered = {
        "DebtToIncome"     : raw["DebtRatio"] * income,
        "IncomePerDependent": income / (deps + 1),
        "TotalPastDue"     : (
            raw["NumberOfTime30-59DaysPastDueNotWorse"]
            + raw["NumberOfTime60-89DaysPastDueNotWorse"]
            + raw["NumberOfTimes90DaysLate"]
        ),
        "UtilRatio_Age"    : (
            raw["RevolvingUtilizationOfUnsecuredLines"] * np.log1p(raw["age"])
        ),
        "HighUtilization"  : int(raw["RevolvingUtilizationOfUnsecuredLines"] > 0.75),
    }
    return engineered


# ─────────────────────────────────────────────────────────────────────
# Collect input values interactively or from demo preset
# ─────────────────────────────────────────────────────────────────────
def collect_inputs(demo: bool = False) -> dict:
    """Prompt user for each feature value, or use demo preset."""
    print("\n" + "═" * 60)
    print("  Credit Scoring — Applicant Input")
    print("═" * 60)

    raw = {}

    if demo:
        print("  [DEMO MODE] Using preset applicant values …\n")
        for key, (label, desc, dtype, example) in FEATURE_META.items():
            raw[key] = example
            print(f"  {label:45s}: {example}")
        return raw

    print("  Enter applicant details below.")
    print("  Press ENTER to accept the default value shown in brackets.\n")

    for key, (label, desc, dtype, example) in FEATURE_META.items():
        while True:
            try:
                prompt = f"  {label} [{example}]\n  ({desc})\n  > "
                user_input = input(prompt).strip()
                if user_input == "":
                    raw[key] = dtype(example)
                else:
                    raw[key] = dtype(user_input)
                print()
                break
            except ValueError:
                print(f"  ✗  Please enter a valid {dtype.__name__} value.\n")

    return raw


# ─────────────────────────────────────────────────────────────────────
# Build feature vector in the same order used during training
# ─────────────────────────────────────────────────────────────────────
def build_feature_vector(raw: dict, feature_names: list) -> np.ndarray:
    """Merge raw + engineered features into an ordered 1-D array."""
    all_features = {**raw, **derive_engineered(raw)}

    vector = []
    for name in feature_names:
        if name not in all_features:
            raise KeyError(
                f"Feature '{name}' expected by model but not provided. "
                "Re-run credit_scoring.py to regenerate the saved model."
            )
        vector.append(float(all_features[name]))

    return np.array(vector, dtype=np.float32).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────
# Load model bundle
# ─────────────────────────────────────────────────────────────────────
def load_bundle(path: str = "saved_model/best_credit_model.pkl") -> dict:
    if not os.path.exists(path):
        print(
            f"\n  ✗  Model file not found: {path}\n"
            "  Please run  credit_scoring.py  first to train and save the model."
        )
        sys.exit(1)
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────
# Prediction & result display
# ─────────────────────────────────────────────────────────────────────
def predict(bundle: dict, X: np.ndarray):
    """Scale and predict; return (label, prob_default)."""
    X_sc = bundle["scaler"].transform(X)
    prob_default = bundle["model"].predict_proba(X_sc)[0, 1]
    label = "NOT CREDITWORTHY ⚠" if prob_default >= 0.50 else "CREDITWORTHY ✔"
    return label, prob_default


def display_result(raw: dict, label: str, prob_default: float, model_name: str):
    width = 60
    print("\n" + "═" * width)
    print("  PREDICTION RESULT")
    print("═" * width)
    print(f"  Model used            : {model_name}")
    print(f"  Probability of default: {prob_default * 100:.2f} %")
    print(f"  Verdict               : {label}")
    print("─" * width)

    # Risk breakdown
    prob_safe = 1 - prob_default
    bar_len   = 30
    safe_fill = int(prob_safe * bar_len)
    risk_fill = bar_len - safe_fill
    bar = "█" * safe_fill + "░" * risk_fill
    print(f"\n  Risk gauge (green = safe, red = risky)")
    print(f"  [{bar}]")
    print(f"  Safe: {prob_safe*100:.1f} %  |  Risky: {prob_default*100:.1f} %")

    # Interpretation
    if prob_default < 0.20:
        tier = "LOW RISK  — Strong candidate for credit."
    elif prob_default < 0.40:
        tier = "MODERATE RISK  — Review additional documents."
    elif prob_default < 0.60:
        tier = "HIGH RISK  — Consider reduced credit limit."
    else:
        tier = "VERY HIGH RISK  — Credit application likely to be declined."

    print(f"\n  Assessment : {tier}")
    print("═" * width + "\n")


# ─────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict credit default risk for a new applicant."
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with preset demo values instead of interactive prompts."
    )
    parser.add_argument(
        "--model", default="saved_model/best_credit_model.pkl",
        help="Path to the saved model bundle (default: saved_model/best_credit_model.pkl)"
    )
    args = parser.parse_args()

    # Load model
    bundle        = load_bundle(args.model)
    feature_names = bundle["feature_names"]
    model_name    = bundle["model_name"]

    print(f"\n  Loaded : {model_name}  ({args.model})")

    # Collect user inputs
    raw = collect_inputs(demo=args.demo)

    # Build feature vector
    try:
        X = build_feature_vector(raw, feature_names)
    except KeyError as e:
        print(f"\n  ✗  {e}")
        sys.exit(1)

    # Predict
    label, prob_default = predict(bundle, X)

    # Display
    display_result(raw, label, prob_default, model_name)


if __name__ == "__main__":
    main()
