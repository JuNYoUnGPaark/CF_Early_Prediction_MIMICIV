# -*- coding: utf-8 -*-
# name: model_training.py
# author: JunYoung Park
# date: 2026-08-28


from pathlib import Path
import itertools
import json
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------------------------------------------------------------------------------------------------
# 1. Load feature data
# -----------------------------------------------------------------------------------------------------------------------
def load_feature_data(csv_path):
    df = pd.read_csv(csv_path)
    df["gridtime"] = pd.to_datetime(df["gridtime"], errors="coerce")

    meta_cols = ["stay_id", "gridtime", "set", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # LightGBM native categorical handling
    categorical_cols = []

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
            categorical_cols.append(col)

    # inf -> NaN
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    df[numeric_cols] = df[numeric_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df, feature_cols, categorical_cols


# -----------------------------------------------------------------------------------------------------------------------
# 2. LightGBM hyperparameter grid
# -----------------------------------------------------------------------------------------------------------------------
def build_hyperparameter_grid():
    grid = {
        "n_estimators": [5000],
        "num_leaves": [8, 16, 32, 64, 128],
        "learning_rate": [0.05],
        "colsample_bytree": [0.33, 0.66],
        "subsample": [0.33, 0.66]
    }

    keys = list(grid.keys())

    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(
            *(grid[key] for key in keys)
        )
    ]

    return combinations


# -----------------------------------------------------------------------------------------------------------------------
# 3. Build LightGBM model
# -----------------------------------------------------------------------------------------------------------------------
def build_lightgbm_model(
    hyperparameters,
    random_state=42
):
    model = LGBMClassifier(
        objective="binary",

        # Hyperparameter search
        n_estimators=hyperparameters["n_estimators"],
        num_leaves=hyperparameters["num_leaves"],
        learning_rate=hyperparameters["learning_rate"],
        colsample_bytree=hyperparameters["colsample_bytree"],
        subsample=hyperparameters["subsample"],

        # Required for row subsampling
        subsample_freq=1,

        # Fixed settings
        is_unbalance=True,
        min_child_samples=1000,
        subsample_for_bin=1_000_000,

        random_state=random_state,
        n_jobs=-1,
        verbosity=-1
    )

    return model


# -----------------------------------------------------------------------------------------------------------------------
# 4. Train one split
# -----------------------------------------------------------------------------------------------------------------------
def train_lightgbm_split(
    csv_path,
    split_id,
    output_dir,
    random_state=42
):
    print(f"\n{'=' * 100}")
    print(f"Split {split_id}")
    print(f"{'=' * 100}")

    df, feature_cols, categorical_cols = load_feature_data(csv_path)

    train = df.loc[df["set"] == "train"].copy()
    val = df.loc[df["set"] == "validation"].copy()
    test = df.loc[df["set"] == "test"].copy()

    X_train = train[feature_cols]
    y_train = train["label"].astype("int8")

    X_val = val[feature_cols]
    y_val = val["label"].astype("int8")

    X_test = test[feature_cols]
    y_test = test["label"].astype("int8")

    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(val):,}")
    print(f"Test       : {len(test):,}")
    print(f"Features   : {len(feature_cols):,}")
    print(f"Categorical: {categorical_cols}")
    print(f"Train positive rate: {y_train.mean() * 100:.2f}%")

    hp_grid = build_hyperparameter_grid()

    print(f"HP combinations: {len(hp_grid)}")

    hp_results = []
    best_model = None
    best_hp = None
    best_val_auprc = -np.inf
    best_iteration = None

    # -------------------------------------------------------------------------------------------------------------------
    # Hyperparameter search
    # -------------------------------------------------------------------------------------------------------------------
    for hp_idx, hp in enumerate(hp_grid, start=1):
        print(
            f"[{hp_idx:02d}/{len(hp_grid)}] "
            f"leaves={hp['num_leaves']} | "
            f"feature_fraction={hp['colsample_bytree']} | "
            f"row_fraction={hp['subsample']}"
        )

        model = build_lightgbm_model(
            hyperparameters=hp,
            random_state=random_state
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            categorical_feature=categorical_cols if categorical_cols else "auto",
            callbacks=[
                early_stopping(
                    stopping_rounds=50,
                    first_metric_only=True,
                    verbose=False
                ),
                log_evaluation(period=0)
            ]
        )

        val_prob = model.predict_proba(
            X_val,
            num_iteration=model.best_iteration_
        )[:, 1]

        val_auroc = roc_auc_score(
            y_val,
            val_prob
        )

        val_auprc = average_precision_score(
            y_val,
            val_prob
        )

        hp_result = {
            "split_id": split_id,
            **hp,
            "best_iteration": int(model.best_iteration_),
            "val_auroc": float(val_auroc),
            "val_auprc": float(val_auprc)
        }

        hp_results.append(hp_result)

        print(
            f"    iteration={model.best_iteration_:4d} | "
            f"Val AUROC={val_auroc:.4f} | "
            f"Val AUPRC={val_auprc:.4f}"
        )

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_model = model
            best_hp = hp.copy()
            best_iteration = int(model.best_iteration_)

    # -------------------------------------------------------------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------------------------------------------------------------
    test_prob = best_model.predict_proba(
        X_test,
        num_iteration=best_iteration
    )[:, 1]

    test_auroc = roc_auc_score(
        y_test,
        test_prob
    )

    test_auprc = average_precision_score(
        y_test,
        test_prob
    )

    print("\n[Best model]")
    print(f"num_leaves       : {best_hp['num_leaves']}")
    print(f"feature_fraction : {best_hp['colsample_bytree']}")
    print(f"row_fraction     : {best_hp['subsample']}")
    print(f"best_iteration   : {best_iteration}")
    print(f"Validation AUPRC : {best_val_auprc:.4f}")
    print(f"Test AUROC       : {test_auroc:.4f}")
    print(f"Test AUPRC       : {test_auprc:.4f}")

    # -------------------------------------------------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hp_results_df = pd.DataFrame(hp_results)

    hp_results_df.to_csv(
        output_dir / f"lightgbm_hp_split_{split_id}.csv",
        index=False
    )

    predictions = test[
        ["stay_id", "gridtime", "label"]
    ].copy()

    predictions["probability"] = test_prob

    predictions.to_csv(
        output_dir / f"lightgbm_predictions_split_{split_id}.csv",
        index=False
    )

    joblib.dump(
        best_model,
        output_dir / f"lightgbm_model_split_{split_id}.pkl"
    )

    with open(
        output_dir / f"lightgbm_best_hp_split_{split_id}.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                **best_hp,
                "best_iteration": best_iteration,
                "validation_auprc": best_val_auprc
            },
            f,
            indent=4
        )

    result = {
        "split_id": split_id,
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(test),
        "n_features": len(feature_cols),
        "best_num_leaves": best_hp["num_leaves"],
        "best_feature_fraction": best_hp["colsample_bytree"],
        "best_row_fraction": best_hp["subsample"],
        "best_iteration": best_iteration,
        "validation_auprc": best_val_auprc,
        "test_auroc": test_auroc,
        "test_auprc": test_auprc
    }

    return result


# -----------------------------------------------------------------------------------------------------------------------
# 5. Run 5 splits
# -----------------------------------------------------------------------------------------------------------------------
def run_lightgbm_training(
    feature_root,
    output_dir,
    n_splits=5
):
    feature_root = Path(feature_root)
    output_dir = Path(output_dir)

    results = []

    for split_id in range(1, n_splits + 1):
        csv_path = (
            feature_root /
            f"features_split_{split_id}.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {csv_path}"
            )

        result = train_lightgbm_split(
            csv_path=csv_path,
            split_id=split_id,
            output_dir=output_dir,
            random_state=42 + split_id
        )

        results.append(result)

    results = pd.DataFrame(results)

    results.to_csv(
        output_dir / "lightgbm_results.csv",
        index=False
    )

    auroc_mean = results["test_auroc"].mean()
    auroc_sd = results["test_auroc"].std(ddof=1)

    auprc_mean = results["test_auprc"].mean()
    auprc_sd = results["test_auprc"].std(ddof=1)

    print(f"\n{'=' * 100}")
    print("[Final LightGBM Result]")
    print(f"{'=' * 100}")

    for row in results.itertuples(index=False):
        print(
            f"Split {row.split_id} | "
            f"AUROC: {row.test_auroc:.4f} | "
            f"AUPRC: {row.test_auprc:.4f}"
        )

    print()
    print(
        f"AUROC : {auroc_mean:.4f} ± {auroc_sd:.4f}"
    )

    print(
        f"AUPRC : {auprc_mean:.4f} ± {auprc_sd:.4f}"
    )

    summary = pd.DataFrame([
        {
            "metric": "AUROC",
            "mean": auroc_mean,
            "sd": auroc_sd
        },
        {
            "metric": "AUPRC",
            "mean": auprc_mean,
            "sd": auprc_sd
        }
    ])

    summary.to_csv(
        output_dir / "lightgbm_summary.csv",
        index=False
    )

    return results, summary