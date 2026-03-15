import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib

from config import PROCESSED_DIR
from spark_pipeline import build_title_features


def vector_column_to_csr(pdf, vector_col, num_features=2000):
    rows = []
    cols = []
    data = []

    for row_idx, vec in enumerate(pdf[vector_col]):
        for col_idx, value in zip(vec.indices, vec.values):
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(value)

    return csr_matrix((data, (rows, cols)), shape=(len(pdf), num_features))


def prepare_numeric_features(df, imputer=None, scaler=None, fit=False):
    feature_cols = [
        "startYear",
        "endYear",
        "runtimeMinutes",
        "numVotes",
        "num_directors",
        "num_writers",
    ]

    X = df[feature_cols].copy()

    if fit:
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

    return X_scaled, imputer, scaler


def evaluate_model(name, X_train, X_valid, y_train, y_valid, estimator):
    """Fit `estimator` on the split and print metrics, then return fitted estimator."""
    clf = estimator
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)

    print(f"\n{name} accuracy:")
    print(accuracy_score(y_valid, preds))

    print(f"\n{name} classification report:")
    print(classification_report(y_valid, preds))
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default=None, help="Path to save trained model (joblib)")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV to tune RF hyperparameters")
    parser.add_argument("--n-iter", type=int, default=30, help="Number of iterations for RandomizedSearchCV")
    args = parser.parse_args()

    train_base = pd.read_parquet(PROCESSED_DIR / "train_features_base.parquet")

    spark, train_title_sdf, _, _ = build_title_features()
    train_title_pdf = train_title_sdf.toPandas()
    spark.stop()

    train_base = train_base.sort_values("tconst").reset_index(drop=True)
    train_title_pdf = train_title_pdf.sort_values("tconst").reset_index(drop=True)

    assert (train_base["tconst"].values == train_title_pdf["tconst"].values).all()

    y = train_base["label"].astype(int).values
    X_title_full = vector_column_to_csr(train_title_pdf, "title_tfidf", num_features=2000)

    indices = np.arange(len(train_base))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = train_base.iloc[train_idx].reset_index(drop=True)
    valid_df = train_base.iloc[valid_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_valid = y[valid_idx]

    X_train_num, imputer, scaler = prepare_numeric_features(train_df, fit=True)
    X_valid_num, _, _ = prepare_numeric_features(valid_df, imputer=imputer, scaler=scaler, fit=False)

    X_train_num = csr_matrix(X_train_num)
    X_valid_num = csr_matrix(X_valid_num)

    X_train_title = X_title_full[train_idx]
    X_valid_title = X_title_full[valid_idx]

    X_train_combined = hstack([X_train_num, X_train_title])
    X_valid_combined = hstack([X_valid_num, X_valid_title])

    # Base RandomForest estimator
    base_rf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42, n_jobs=-1)

    if args.tune:
        print("Running hyperparameter search (RandomizedSearchCV)...")
        param_dist = {
            "n_estimators": [100, 200, 400, 800],
            "max_depth": [5, 10, 20, None],
            "max_features": ["sqrt", "log2", 0.2, 0.5],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        }

        rs = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="accuracy",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        # Tune on the combined training folds
        rs.fit(X_train_combined, y_train)
        print(f"Best params: {rs.best_params_}")
        print(f"Best CV score: {rs.best_score_}")
        best_rf = rs.best_estimator_
    else:
        best_rf = base_rf

    evaluate_model("Numeric only", X_train_num, X_valid_num, y_train, y_valid, estimator=best_rf)
    evaluate_model("Title only", X_train_title, X_valid_title, y_train, y_valid, estimator=best_rf)
    clf = evaluate_model("Combined", X_train_combined, X_valid_combined, y_train, y_valid, estimator=best_rf)

    # Persist the trained combined model and preprocessors
    models_dir = Path(__file__).resolve().parent.parent / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # determine save path
    model_tag = "rf_tuned" if getattr(args, "tune", False) else "rf"
    if args.save is None:
        save_path = models_dir / f"combined_{model_tag}_model.joblib"
    else:
        save_path = Path(args.save)

    joblib.dump({"model": clf, "imputer": imputer, "scaler": scaler}, save_path)
    print(f"\nSaved model and preprocessors to: {save_path}")


if __name__ == "__main__":
    main()