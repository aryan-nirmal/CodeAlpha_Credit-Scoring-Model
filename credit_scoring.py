"""
╔══════════════════════════════════════════════════════════════════════╗
║         Credit Scoring Model — CodeAlpha ML Internship Task 1        ║
║  Dataset : Give Me Some Credit (Kaggle)                               ║
║  Models  : Logistic Regression | Random Forest | XGBoost             ║
║  Author  : CodeAlpha Intern                                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────
# 0.  Imports & global settings
# ─────────────────────────────────────────────────────────────────────
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────
PLOTS_DIR = "plots"
MODEL_DIR  = "saved_model"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ════════════════════════════════════════════════════════════════════
def load_data(filepath: str = "cs-training.csv") -> pd.DataFrame:
    """Load the Give Me Some Credit dataset."""
    print(f"\n{'='*60}")
    print("  STEP 1 — Loading data")
    print(f"{'='*60}")

    df = pd.read_csv(filepath, index_col=0)
    print(f"  Shape           : {df.shape}")
    print(f"  Target balance  :\n{df['SeriousDlqin2yrs'].value_counts()}")
    return df


# ════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new informative features from the raw columns.

    New features
    ────────────
    DebtToIncome        : MonthlyIncome / (DebtRatio * MonthlyIncome + 1)
    IncomePerDependent  : MonthlyIncome / (NumberOfDependents + 1)
    TotalPastDue        : sum of all past-due occurrence columns
    UtilRatio_Age       : RevolvingUtilization × log1p(age) — stress-age interaction
    HighUtilization     : binary flag (RevolvingUtilization > 0.75)
    """
    print(f"\n{'='*60}")
    print("  STEP 2 — Feature engineering")
    print(f"{'='*60}")

    df = df.copy()

    # Debt-to-income proxy
    df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"].fillna(
        df["MonthlyIncome"].median()
    )

    # Income per dependent (avoid div-by-zero)
    df["IncomePerDependent"] = (
        df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
        / (df["NumberOfDependents"].fillna(0) + 1)
    )

    # Total past-due events across all time windows
    past_due_cols = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df["TotalPastDue"] = df[past_due_cols].sum(axis=1)

    # Utilization × age interaction
    df["UtilRatio_Age"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] * np.log1p(df["age"])
    )

    # Binary high-utilization flag
    df["HighUtilization"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] > 0.75
    ).astype(int)

    print(f"  Features after engineering : {df.shape[1]-1}")
    return df


# ════════════════════════════════════════════════════════════════════
# 3.  DATA PREPROCESSING
# ════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    """
    Steps
    ─────
    • Impute missing values (median strategy)
    • Cap extreme outliers at 99th percentile for selected columns
    • Split → train / test (stratified 80/20)
    • Scale features
    • Apply SMOTE to training set
    """
    print(f"\n{'='*60}")
    print("  STEP 3 — Preprocessing")
    print(f"{'='*60}")

    TARGET = "SeriousDlqin2yrs"
    FEATURES = [c for c in df.columns if c != TARGET]

    # ── 3a. Impute ──────────────────────────────────────────────────
    print("  Imputing missing values …")
    for col in df[FEATURES].columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    {col:45s} → median = {median_val:.2f}")

    # ── 3b. Outlier capping ─────────────────────────────────────────
    print("  Capping outliers at 99th percentile …")
    cap_cols = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "TotalPastDue",
        "DebtToIncome",
        "IncomePerDependent",
    ]
    for col in cap_cols:
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=p99)

    # ── 3c. Split ───────────────────────────────────────────────────
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")

    # ── 3d. Scale ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── 3e. SMOTE ───────────────────────────────────────────────────
    print("  Applying SMOTE to training set …")
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_sc, y_train)
    unique, counts = np.unique(y_train_bal, return_counts=True)
    print(f"  Post-SMOTE class distribution : {dict(zip(unique, counts))}")

    return (
        X_train_bal, X_test_sc,
        y_train_bal, y_test,
        scaler, FEATURES,
    )


# ════════════════════════════════════════════════════════════════════
# 4.  CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df: pd.DataFrame):
    """Save a correlation heatmap for all numeric features."""
    print("  Saving correlation heatmap …")

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=0.6, center=0,
        annot=True, fmt=".2f", linewidths=0.4,
        square=True, ax=ax, annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    → {path}")


# ════════════════════════════════════════════════════════════════════
# 5.  MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════
def build_models() -> dict:
    """Return a dict of {name: estimator}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, C=0.5
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=13,           # ~ ratio of majority/minority
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42, n_jobs=-1,
        ),
    }


# ════════════════════════════════════════════════════════════════════
# 6.  TRAINING & EVALUATION
# ════════════════════════════════════════════════════════════════════
def train_and_evaluate(
    models: dict,
    X_train, X_test,
    y_train, y_test,
    feature_names: list,
) -> tuple[dict, dict, str]:
    """
    Train every model, compute metrics, and collect predictions.

    Returns
    ───────
    results      : {model_name: metrics_dict}
    predictions  : {model_name: (y_pred, y_proba)}
    best_name    : name of the model with highest ROC-AUC
    """
    print(f"\n{'='*60}")
    print("  STEP 4 — Training & evaluation")
    print(f"{'='*60}")

    results     = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n  ── {name} ──────────────────────────────────")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy" : accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall"   : recall_score(y_test, y_pred, zero_division=0),
            "F1-Score" : f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC"  : roc_auc_score(y_test, y_proba),
        }

        for k, v in metrics.items():
            print(f"    {k:12s}: {v:.4f}")

        results[name]     = metrics
        predictions[name] = (y_pred, y_proba)

        # Classification report
        print(f"\n  Classification Report — {name}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No Default", "Default"]))

    # Best by ROC-AUC
    best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    print(f"\n  ★  Best model (ROC-AUC) : {best_name}  "
          f"(AUC = {results[best_name]['ROC-AUC']:.4f})")

    return results, predictions, best_name


# ════════════════════════════════════════════════════════════════════
# 7.  PLOTS
# ════════════════════════════════════════════════════════════════════
def plot_roc_curves(models, predictions, y_test):
    """All three ROC curves on one chart."""
    print("  Saving ROC curves …")

    colours = ["#2563EB", "#16A34A", "#DC2626"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")

    for (name, (_, y_proba)), colour in zip(predictions.items(), colours):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colour, lw=2, label=f"{name}  (AUC = {auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    → {path}")


def plot_confusion_matrices(predictions, y_test):
    """One confusion-matrix subplot per model."""
    print("  Saving confusion matrices …")

    names = list(predictions.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, name in zip(axes, names):
        y_pred, _ = predictions[name]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path}")


def plot_feature_importance(models, feature_names):
    """
    Feature importance for tree-based models.
    Coefficient magnitudes for Logistic Regression.
    """
    print("  Saving feature importance plots …")

    tree_models = {
        name: m for name, m in models.items()
        if hasattr(m, "feature_importances_")
    }

    n = len(tree_models) + 1           # +1 for LR
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))

    # Logistic Regression — coefficient magnitudes
    lr_model = models["Logistic Regression"]
    coef = np.abs(lr_model.coef_[0])
    order = np.argsort(coef)[-15:]     # top 15
    axes[0].barh(
        [feature_names[i] for i in order], coef[order],
        color="#2563EB", edgecolor="white"
    )
    axes[0].set_title("Logistic Regression\n|Coefficients|", fontweight="bold")
    axes[0].set_xlabel("Magnitude")

    # Tree-based models — Gini importance
    colours = ["#16A34A", "#DC2626"]
    for ax, (name, model), colour in zip(axes[1:], tree_models.items(), colours):
        imp   = model.feature_importances_
        order = np.argsort(imp)[-15:]
        ax.barh(
            [feature_names[i] for i in order], imp[order],
            color=colour, edgecolor="white"
        )
        ax.set_title(f"{name}\nFeature Importances", fontweight="bold")
        ax.set_xlabel("Importance")

    fig.suptitle("Top Feature Importances — All Models",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {path}")


def plot_metrics_comparison(results: dict):
    """Grouped bar chart comparing all metrics across models."""
    print("  Saving metrics comparison chart …")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.22
    colours = ["#2563EB", "#16A34A", "#DC2626"]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, colour) in enumerate(zip(model_names, colours)):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=colour, edgecolor="white", alpha=0.87)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "metrics_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    → {path}")


# ════════════════════════════════════════════════════════════════════
# 8.  SAVE BEST MODEL
# ════════════════════════════════════════════════════════════════════
def save_best_model(models, best_name, scaler, feature_names):
    """Persist the best model + scaler as a single bundle."""
    bundle = {
        "model"        : models[best_name],
        "scaler"       : scaler,
        "feature_names": feature_names,
        "model_name"   : best_name,
    }
    path = os.path.join(MODEL_DIR, "best_credit_model.pkl")
    joblib.dump(bundle, path, compress=3)
    print(f"\n  ✔  Best model saved → {path}")


# ════════════════════════════════════════════════════════════════════
# 9.  RESULTS TABLE
# ════════════════════════════════════════════════════════════════════
def print_results_table(results: dict):
    df = pd.DataFrame(results).T
    df = df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    df = df.round(4)
    print(f"\n{'='*60}")
    print("  FINAL RESULTS TABLE")
    print(f"{'='*60}")
    print(df.to_string())
    print(f"{'='*60}\n")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 60)
    print("  Credit Scoring Model — CodeAlpha ML Internship")
    print("█" * 60)

    # 1. Load
    df = load_data("cs-training.csv")

    # 2. Correlation heatmap on raw data (before engineering)
    print(f"\n{'='*60}")
    print("  STEP 2a — Correlation heatmap (raw features)")
    print(f"{'='*60}")
    plot_correlation_heatmap(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # 5. Build models
    models = build_models()

    # 6. Train & evaluate
    results, predictions, best_name = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, feature_names
    )

    # 7. Plots
    print(f"\n{'='*60}")
    print("  STEP 5 — Generating plots")
    print(f"{'='*60}")
    plot_roc_curves(models, predictions, y_test)
    plot_confusion_matrices(predictions, y_test)
    plot_feature_importance(models, feature_names)
    plot_metrics_comparison(results)

    # 8. Save best model
    save_best_model(models, best_name, scaler, feature_names)

    # 9. Print results table
    print_results_table(results)

    print("  ✔  All done! Check plots/ and saved_model/ directories.\n")


if __name__ == "__main__":
    main()
